use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use rayon::prelude::*;
use rust_llm_stock::{stock_db::get_connection, ts::model::DailyModel};
use sqlx::{Pool, Postgres};
use tokio::task;

/// Struct to hold adjusted daily record for batch insertion
#[derive(Clone, Debug)]
struct AdjustedDailyRecord {
    ts_code: String,
    trade_date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    amount: Option<f64>,
    pct_chg: Option<f64>,
    adj_factor: f64,
}

/// Result struct for processing a stock
#[derive(Debug)]
struct ProcessResult {
    ts_code: String,
    inserted_count: usize,
    skipped: bool,
    adjustment_event: bool,
}

/// Calculate adjustment factors based on pre_close vs previous day's close
/// Returns the actual adjustment ratio for each day (not cumulative)
fn calculate_adjustment_factors(daily_data: &[DailyModel]) -> HashMap<String, f64> {
    let mut adj_factors = HashMap::new();

    for (i, current) in daily_data.iter().enumerate() {
        if let Some(ref trade_date) = current.trade_date {
            if i == 0 {
                // First trading day always has factor 1.0
                adj_factors.insert(trade_date.clone(), 1.0);
            } else {
                // Check if there's a split/dividend by comparing pre_close with prev close
                if let (Some(curr_pre_close), Some(prev_close)) =
                    (current.pre_close, daily_data[i - 1].close)
                {
                    // If pre_close differs from previous day's close, it indicates an adjustment
                    if (curr_pre_close - prev_close).abs() > 0.0001 {
                        // Save the actual adjustment ratio: pre_close / prev_close
                        let adjustment_ratio = curr_pre_close / prev_close;
                        adj_factors.insert(trade_date.clone(), adjustment_ratio);
                    } else {
                        // No adjustment happened, factor = 1.0
                        adj_factors.insert(trade_date.clone(), 1.0);
                    }
                } else {
                    adj_factors.insert(trade_date.clone(), 1.0);
                }
            }
        }
    }

    adj_factors
}

/// Process a single stock and return structured result
/// Note: pool is Arc for sharing across async/parallel contexts
async fn process_stock(
    pool: Arc<Pool<Postgres>>,
    ts_code: String,
    list_date: String,
) -> Result<ProcessResult, Box<dyn Error + Send + Sync>> {
    // Check if this stock needs updating (incremental backfill)
    let max_adjusted_date: Option<String> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM adjusted_stock_daily WHERE ts_code = $1")
            .bind(&ts_code)
            .fetch_optional(pool.as_ref())
            .await?;

    let max_daily_date: Option<String> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM stock_daily WHERE ts_code = $1")
            .bind(&ts_code)
            .fetch_optional(pool.as_ref())
            .await?;

    // Check for adjustment events in new data (pre_close != previous day's close)
    // If there's an adjustment event, we MUST recalculate the entire history
    let has_adjustment_event = if max_adjusted_date.is_some() {
        let adjustment_check: Option<(i64,)> = sqlx::query_as(
            r#"
            WITH daily_with_prev AS (
                SELECT 
                    trade_date,
                    close,
                    pre_close,
                    LAG(close) OVER (ORDER BY trade_date) as prev_close
                FROM stock_daily
                WHERE ts_code = $1 
                  AND trade_date > $2
                ORDER BY trade_date
            )
            SELECT COUNT(*)
            FROM daily_with_prev
            WHERE ABS(COALESCE(pre_close, 0) - COALESCE(prev_close, 0)) > 0.0001
            "#,
        )
        .bind(&ts_code)
        .bind(&max_adjusted_date)
        .fetch_optional(pool.as_ref())
        .await?;

        adjustment_check.map(|(count,)| count > 0).unwrap_or(false)
    } else {
        false // No existing data, will process full history anyway
    };

    // Skip if already up-to-date AND no adjustment events
    if max_adjusted_date.is_some() && max_adjusted_date == max_daily_date && !has_adjustment_event {
        return Ok(ProcessResult {
            ts_code,
            inserted_count: 0,
            skipped: true,
            adjustment_event: false,
        });
    }

    // Fetch all daily data from listing date (need full history for adjustment factors)
    let query = format!(
        r#"
        SELECT ts_code, trade_date, open, high, low, close, pre_close, 
               vol, amount, 
               change, pct_chg
        FROM stock_daily 
        WHERE ts_code = $1 AND trade_date >= '{}'
        ORDER BY trade_date ASC
        "#,
        list_date
    );

    let daily_data = sqlx::query_as::<_, DailyModel>(&query)
        .bind(&ts_code)
        .fetch_all(pool.as_ref())
        .await?;

    if daily_data.is_empty() {
        return Ok(ProcessResult {
            ts_code,
            inserted_count: 0,
            skipped: true,
            adjustment_event: false,
        });
    }

    // Calculate adjustment factors (now returns actual shift ratios, not cumulative)
    let adj_factors = calculate_adjustment_factors(&daily_data);

    // Collect records for batch insertion
    let mut records = Vec::with_capacity(daily_data.len());

    // Accumulate adjustment factors backward (from earliest to latest)
    let mut cumulative_forward = Vec::new();
    let mut cumulative = 1.0;

    for day in daily_data.iter() {
        if let Some(ref trade_date) = day.trade_date {
            let adj_factor = adj_factors.get(trade_date).copied().unwrap_or(1.0);
            cumulative_forward.push(cumulative);
            if (adj_factor - 1.0).abs() > 0.0001 {
                cumulative *= adj_factor;
            }
        } else {
            cumulative_forward.push(1.0);
        }
    }

    // Calculate the final cumulative factor (at the most recent date)
    let final_cumulative = cumulative;

    // Now process each day with backward adjustment
    for (i, day) in daily_data.iter().enumerate() {
        if day.open.is_none()
            || day.high.is_none()
            || day.low.is_none()
            || day.close.is_none()
            || day.vol.is_none()
        {
            continue;
        }

        // Get raw prices
        let raw_open = day.open.unwrap();
        let raw_high = day.high.unwrap();
        let raw_low = day.low.unwrap();
        let raw_close = day.close.unwrap();
        let volume = day.vol.unwrap();
        let amount = day.amount;
        let pct_chg = day.pct_chg;

        // Apply backward adjustment: multiply by (final_cumulative / current_cumulative)
        let current_cumulative = cumulative_forward[i];
        let backward_adj_factor = final_cumulative / current_cumulative;

        let adjusted_open = (raw_open * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_high = (raw_high * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_low = (raw_low * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_close = (raw_close * backward_adj_factor * 100.0).round() / 100.0;

        // Get the adjustment factor for this day
        let adj_factor = if let Some(ref trade_date) = day.trade_date {
            adj_factors.get(trade_date).copied().unwrap_or(1.0)
        } else {
            1.0
        };

        records.push(AdjustedDailyRecord {
            ts_code: ts_code.clone(),
            trade_date: day.trade_date.clone().unwrap_or_default(),
            open: adjusted_open,
            high: adjusted_high,
            low: adjusted_low,
            close: adjusted_close,
            volume,
            amount,
            pct_chg,
            adj_factor,
        });
    }

    // Batch insert all records for this stock
    let inserted_count = batch_insert_adjusted_daily(pool.as_ref(), &records).await?;

    Ok(ProcessResult {
        ts_code,
        inserted_count,
        skipped: false,
        adjustment_event: has_adjustment_event,
    })
}

/// Batch insert adjusted daily records using transactions for performance
async fn batch_insert_adjusted_daily(
    pool: &Pool<Postgres>,
    records: &[AdjustedDailyRecord],
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    if records.is_empty() {
        return Ok(0);
    }

    let mut tx = pool.begin().await?;
    let mut total_inserted = 0;

    // Process in chunks for better performance while avoiding query size limits
    let chunk_size = 50;

    for chunk in records.chunks(chunk_size) {
        for record in chunk {
            sqlx::query(
                r#"
                INSERT INTO adjusted_stock_daily (
                    ts_code, trade_date, open, high, low, close, 
                    volume, amount, pct_chg, adj_factor
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                )
                ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    amount = EXCLUDED.amount,
                    pct_chg = EXCLUDED.pct_chg,
                    adj_factor = EXCLUDED.adj_factor
                "#,
            )
            .bind(&record.ts_code)
            .bind(&record.trade_date)
            .bind(record.open)
            .bind(record.high)
            .bind(record.low)
            .bind(record.close)
            .bind(record.volume)
            .bind(record.amount)
            .bind(record.pct_chg)
            .bind(record.adj_factor)
            .execute(&mut *tx)
            .await?;

            total_inserted += 1;
        }
    }

    tx.commit().await?;
    Ok(total_inserted)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let pool = Arc::new(get_connection().await);

    // Get all stocks from stock_basic table (only listed stocks)
    let stocks = sqlx::query_as::<_, (String, String)>(
        r#"
        SELECT sb.ts_code, sb.list_date
        FROM stock_basic sb
        WHERE sb.list_date IS NOT NULL
        ORDER BY sb.ts_code
        "#,
    )
    .fetch_all(pool.as_ref())
    .await?;

    println!(
        "\n🚀 Processing {} stocks with connection pooling...\n",
        stocks.len()
    );

    let start_time = std::time::Instant::now();

    // Convert to Arc-wrapped for sharing in parallel context
    let stocks_arc = Arc::new(stocks);
    let total_stocks = stocks_arc.len();

    // Process in batches to avoid connection pool exhaustion
    // With 64 connections in pool, using batch size of 16 ensures we never wait for connections
    const BATCH_SIZE: usize = 16;
    let mut total_records = 0;
    let mut successful_stocks = 0;

    for batch_start in (0..total_stocks).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(total_stocks);
        let mut handles = vec![];

        // Spawn tasks for this batch
        for idx in batch_start..batch_end {
            let (ts_code, list_date) = &stocks_arc[idx];
            let pool_clone = Arc::clone(&pool);
            let ts_code_clone = ts_code.clone();
            let list_date_clone = list_date.clone();

            let handle = tokio::spawn(async move {
                match process_stock(pool_clone, ts_code_clone.clone(), list_date_clone).await {
                    Ok(result) => {
                        if result.skipped {
                            eprintln!(
                                "[{}/{}] ⏭  {}: already up-to-date",
                                idx + 1,
                                total_stocks,
                                ts_code_clone
                            );
                        } else {
                            println!(
                                "[{}/{}] ✓ {}: {} records{}",
                                idx + 1,
                                total_stocks,
                                ts_code_clone,
                                result.inserted_count,
                                if result.adjustment_event {
                                    " (adjustment event detected)"
                                } else {
                                    ""
                                }
                            );
                        }
                        Some((result.ts_code, result.inserted_count))
                    }
                    Err(e) => {
                        eprintln!("[{}/{}] ✗ {}: {}", idx + 1, total_stocks, ts_code_clone, e);
                        None
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks in this batch to complete before starting next batch
        for handle in handles {
            if let Ok(Some((_ts_code, count))) = handle.await {
                total_records += count;
                if count > 0 {
                    successful_stocks += 1;
                }
            }
        }
    }

    let elapsed = start_time.elapsed();

    println!("\n📊 Summary:");
    println!("  - Total stocks: {}", total_stocks);
    println!("  - Successful: {} stocks", successful_stocks);
    println!("  - Total records inserted: {}", total_records);
    println!("  - Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  - Average: {:.3}s per stock",
        elapsed.as_secs_f64() / total_stocks as f64
    );
    println!("  - Batch size: {} (connection pool: 64)", BATCH_SIZE);

    Ok(())
}
