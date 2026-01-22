use std::collections::HashMap;
use std::error::Error;

use rust_llm_stock::{stock_db::get_connection, ts::model::DailyModel};
use sqlx::{Pool, Postgres};

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

/// Find stocks with new adjustment events (where pre_close != previous day's close)
async fn find_stocks_with_adjustments(
    pool: &Pool<Postgres>,
    lookback_days: i32,
) -> Result<Vec<String>, Box<dyn Error>> {
    let query = format!(
        r#"
        WITH recent_data AS (
            SELECT 
                sd1.ts_code,
                sd1.trade_date,
                sd1.pre_close,
                sd1.close,
                LAG(sd1.close) OVER (PARTITION BY sd1.ts_code ORDER BY sd1.trade_date) as prev_close
            FROM stock_daily sd1
            WHERE sd1.trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '{} days', 'YYYYMMDD')
        )
        SELECT DISTINCT ts_code
        FROM recent_data
        WHERE pre_close IS NOT NULL 
          AND prev_close IS NOT NULL
          AND ABS(pre_close - prev_close) > 0.0001
        ORDER BY ts_code
        "#,
        lookback_days
    );

    let stocks: Vec<(String,)> = sqlx::query_as(&query).fetch_all(pool).await?;

    Ok(stocks.into_iter().map(|(ts_code,)| ts_code).collect())
}

/// Process a single stock - recalculate all adjusted prices
async fn process_stock(
    pool: &Pool<Postgres>,
    ts_code: &str,
    list_date: &str,
) -> Result<usize, Box<dyn Error>> {
    // Fetch all daily data from listing date
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
        .bind(ts_code)
        .fetch_all(pool)
        .await?;

    if daily_data.is_empty() {
        return Ok(0);
    }

    // Calculate adjustment factors
    let adj_factors = calculate_adjustment_factors(&daily_data);

    let mut tx = pool.begin().await?;
    let mut updated_count = 0;

    // Build cumulative factors forward
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

    let final_cumulative = cumulative;

    // Process each day with backward adjustment
    for (i, day) in daily_data.iter().enumerate() {
        if day.open.is_none()
            || day.high.is_none()
            || day.low.is_none()
            || day.close.is_none()
            || day.vol.is_none()
        {
            continue;
        }

        let raw_open = day.open.unwrap();
        let raw_high = day.high.unwrap();
        let raw_low = day.low.unwrap();
        let raw_close = day.close.unwrap();
        let volume = day.vol.unwrap();
        let amount = day.amount;
        let pct_chg = day.pct_chg;

        // Apply backward adjustment
        let current_cumulative = cumulative_forward[i];
        let backward_adj_factor = final_cumulative / current_cumulative;

        let adjusted_open = (raw_open * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_high = (raw_high * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_low = (raw_low * backward_adj_factor * 100.0).round() / 100.0;
        let adjusted_close = (raw_close * backward_adj_factor * 100.0).round() / 100.0;

        let adj_factor = if let Some(ref trade_date) = day.trade_date {
            adj_factors.get(trade_date).copied().unwrap_or(1.0)
        } else {
            1.0
        };

        // Upsert data
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
        .bind(ts_code)
        .bind(&day.trade_date)
        .bind(adjusted_open)
        .bind(adjusted_high)
        .bind(adjusted_low)
        .bind(adjusted_close)
        .bind(volume)
        .bind(amount)
        .bind(pct_chg)
        .bind(adj_factor)
        .execute(&mut *tx)
        .await?;

        updated_count += 1;
    }

    tx.commit().await?;
    Ok(updated_count)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let pool = get_connection().await;

    // Get lookback days from command line arg (default: 7)
    let lookback_days: i32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    println!("\n=== Updating Adjusted Stock Daily for Stocks with Adjustment Events ===");
    println!("Lookback period: {} days\n", lookback_days);

    // Find stocks with adjustment events in recent days
    let stocks_with_adjustments = find_stocks_with_adjustments(&pool, lookback_days).await?;

    if stocks_with_adjustments.is_empty() {
        println!(
            "✓ No adjustment events detected in the last {} days",
            lookback_days
        );
        println!("✓ adjusted_stock_daily is up to date");
        return Ok(());
    }

    println!(
        "Found {} stocks with adjustment events in the last {} days:\n",
        stocks_with_adjustments.len(),
        lookback_days
    );

    // Get list_date for each stock (only listed stocks)
    let stocks_query = format!(
        r#"
        SELECT ts_code, list_date
        FROM stock_basic
        WHERE ts_code = ANY($1) AND list_status = 'L'
        ORDER BY ts_code
        "#
    );

    let stocks: Vec<(String, String)> = sqlx::query_as(&stocks_query)
        .bind(&stocks_with_adjustments)
        .fetch_all(&pool)
        .await?;

    let start_time = std::time::Instant::now();
    let mut total_records = 0;

    for (idx, (ts_code, list_date)) in stocks.iter().enumerate() {
        match process_stock(&pool, ts_code, list_date).await {
            Ok(count) => {
                total_records += count;
                println!(
                    "[{}/{}] ✓ {}: {} records updated (listed: {})",
                    idx + 1,
                    stocks.len(),
                    ts_code,
                    count,
                    list_date
                );
            }
            Err(e) => {
                eprintln!(
                    "[{}/{}] ✗ {}: Error - {}",
                    idx + 1,
                    stocks.len(),
                    ts_code,
                    e
                );
            }
        }
    }

    let elapsed = start_time.elapsed();

    println!("\n=== Update Complete ===");
    println!("✓ Stocks processed: {}", stocks.len());
    println!("✓ Total records updated: {}", total_records);
    println!("✓ Time elapsed: {:.2?}", elapsed);

    Ok(())
}
