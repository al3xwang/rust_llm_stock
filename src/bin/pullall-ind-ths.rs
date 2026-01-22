// Incremental ingestion for industry money flow data from Tushare (moneyflow_ind_ths)

use chrono::{Datelike, Duration as ChronoDuration, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use reqwest;
use rust_llm_stock::stock_db::{create_req, get_connection};
use serde_json;
use sqlx::{Pool, Postgres};
use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());

    println!("Connecting to database...");
    let pool = get_connection().await;
    println!("Connected to database");

    // --- Accept optional start_date from command line ---
    let args: Vec<String> = std::env::args().collect();
    let user_start_date = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    // Get the latest date from moneyflow_ind_ths table, or use default start date
    let latest_date: Option<String> = match sqlx::query_scalar::<_, Option<String>>(
        "SELECT MAX(trade_date) FROM moneyflow_ind_ths WHERE trade_date ~ '^[0-9]{8}$'",
    )
    .fetch_optional(&pool)
    .await
    {
        Ok(result) => result.flatten(),
        Err(e) => {
            eprintln!("Database query error: {}", e);
            None
        }
    };

    let start_date = if let Some(user_date) = user_start_date {
        user_date
    } else if let Some(latest) = latest_date {
        // Start from the day after the latest date
        let date = NaiveDate::parse_from_str(&latest, "%Y%m%d")?;
        let next_day = date + ChronoDuration::days(1);
        next_day.format("%Y%m%d").to_string()
    } else {
        // No data exists, use default start date
        println!("No data in DB, starting from default: 20150101");
        "20150101".to_string()
    };

    let today = Local::now().date_naive();
    let end_date = today.format("%Y%m%d").to_string();

    println!(
        "Pulling industry money flow data from {} to {}",
        start_date, end_date
    );

    let start = NaiveDate::parse_from_str(&start_date, "%Y%m%d")?;

    if start > today {
        println!("No new data to pull (already up to date)");
        return Ok(());
    }

    let total_months =
        (today.year() - start.year()) * 12 + (today.month() as i32 - start.month() as i32) + 1;
    println!(
        "Processing approximately {} months of data",
        total_months.max(1)
    );

    let mut current = start;
    let mut processed_months = 0;

    while current <= today {
        // Wait for rate limiter
        while limiter.check().is_err() {
            thread::sleep(Duration::from_millis(100));
        }

        processed_months += 1;

        let start_date_month = current.format("%Y%m%d").to_string();
        // Calculate end of month or today, whichever is earlier
        let mut next_month = current.with_day(1).unwrap().succ_opt().unwrap();
        next_month = if current.month() == 12 {
            NaiveDate::from_ymd_opt(current.year() + 1, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1).unwrap()
        };
        let end_date_month = std::cmp::min(next_month.pred_opt().unwrap(), today);
        let end_date_str = end_date_month.format("%Y%m%d").to_string();

        println!(
            "[{}/{}] Pulling industry moneyflow from {} to {}",
            processed_months,
            total_months.max(1),
            start_date_month,
            end_date_str
        );

        match ingest_industry_moneyflow(&pool, &start_date_month, &end_date_str).await {
            Ok(count) => {
                if count > 0 {
                    println!("  ✓ Inserted/updated {} records", count);
                } else {
                    println!("  - No data for this period");
                }
            }
            Err(e) => {
                eprintln!("  ✗ Error processing period: {}", e);
            }
        }

        // Move to next month
        current = next_month;
    }

    println!("\n=== Completed ===");
    println!(
        "Processed {} months from {} to {}",
        processed_months, start_date, end_date
    );

    Ok(())
}

async fn ingest_industry_moneyflow(
    pool: &Pool<Postgres>,
    start_date: &str,
    end_date: &str,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let mut params = HashMap::new();
    params.insert("start_date".to_string(), start_date.to_string());
    params.insert("end_date".to_string(), end_date.to_string());

    let request_body = create_req("moneyflow_ind_ths".to_string(), params);

    let client = reqwest::Client::new();
    let api_response = client
        .post("http://api.tushare.pro")
        .body(request_body)
        .send()
        .await?;

    if !api_response.status().is_success() {
        eprintln!(
            "  ✗ API request failed with status: {}",
            api_response.status()
        );
        return Ok(0);
    }

    let response_body = api_response.text().await?;

    let repx: rust_llm_stock::ts::http::Responsex = match serde_json::from_str(&response_body) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ✗ Failed to parse response: {}", e);
            return Ok(0);
        }
    };

    if repx.code != 0 {
        eprintln!("  ✗ API error: {}", repx.msg);
        return Ok(0);
    }

    let items = &repx.data.items;

    if items.is_empty() {
        return Ok(0);
    }

    let mut total_inserted = 0;

    // Map fields to indices dynamically
    let fields = &repx.data.fields;
    let idx_ts_code = fields.iter().position(|f| f == "ts_code");
    let idx_trade_date = fields.iter().position(|f| f == "trade_date");
    let idx_industry = fields.iter().position(|f| f == "industry");
    let idx_net_buy_amount = fields.iter().position(|f| f == "net_buy_amount");
    let idx_net_sell_amount = fields.iter().position(|f| f == "net_sell_amount");
    let idx_net_amount = fields.iter().position(|f| f == "net_amount");

    for item in items.iter() {
        // Only insert if all required indices exist and item has enough columns
        if let (Some(i_ts), Some(i_date), Some(i_ind), Some(i_buy), Some(i_sell), Some(i_net)) = (
            idx_ts_code,
            idx_trade_date,
            idx_industry,
            idx_net_buy_amount,
            idx_net_sell_amount,
            idx_net_amount,
        ) {
            if item.len() > i_net {
                let ts_code = item[i_ts].as_str();
                let trade_date = item[i_date].as_str();
                let industry_name = item[i_ind].as_str();
                let net_buy_amount = item[i_buy].as_f64();
                let net_sell_amount = item[i_sell].as_f64();
                let net_amount = item[i_net].as_f64();

                let insert_result = sqlx::query(
                    r#"
                    INSERT INTO moneyflow_ind_ths (
                        ts_code, trade_date, industry_name, net_buy_amount, net_sell_amount, net_amount
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (ts_code, trade_date) DO NOTHING
                    "#,
                )
                .bind(ts_code)
                .bind(trade_date)
                .bind(industry_name)
                .bind(net_buy_amount)
                .bind(net_sell_amount)
                .bind(net_amount)
                .execute(pool)
                .await;

                match insert_result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            total_inserted += 1;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "    Error inserting industry moneyflow for {} on {}: {}",
                            ts_code.unwrap_or("N/A"),
                            trade_date.unwrap_or("N/A"),
                            e
                        );
                    }
                }
            }
        }
    }

    Ok(total_inserted)
}
