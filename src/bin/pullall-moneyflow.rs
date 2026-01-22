use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Duration as ChronoDuration, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::http::Responsex,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenv::dotenv().ok();

    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());

    println!("Connecting to database...");
    let dbpool = get_connection().await;
    println!("Connected to database");

    // --- Accept optional start_date from command line ---
    let args: Vec<String> = std::env::args().collect();
    let user_start_date = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    // Get the latest date from moneyflow table, or use default start date
    let latest_date: Option<String> = match sqlx::query_scalar::<_, Option<String>>(
        "SELECT MAX(trade_date) FROM moneyflow WHERE trade_date ~ '^[0-9]{8}$'",
    )
    .fetch_optional(&dbpool)
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

    let end_date = Local::now().format("%Y%m%d").to_string();

    println!(
        "Pulling money flow data from {} to {}",
        start_date, end_date
    );

    // Parse dates for iteration
    let start = NaiveDate::parse_from_str(&start_date, "%Y%m%d")?;
    let end = NaiveDate::parse_from_str(&end_date, "%Y%m%d")?;

    if start > end {
        println!("No new data to pull (already up to date)");
        return Ok(());
    }

    let total_days = (end - start).num_days() + 1;
    println!("Processing {} days of data", total_days);

    let mut current_date = start;
    let mut processed_days = 0;

    while current_date <= end {
        // Rate limiting
        loop {
            if limiter.check().is_ok() {
                break;
            } else {
                thread::sleep(Duration::from_millis(100));
            }
        }

        processed_days += 1;
        let trade_date_str = current_date.format("%Y%m%d").to_string();

        println!(
            "[{}/{}] Processing date: {}",
            processed_days, total_days, trade_date_str
        );

        match pull_insert_by_date(&dbpool, trade_date_str.clone()).await {
            Ok(count) => {
                if count > 0 {
                    println!("  ✓ Inserted {} records for {}", count, trade_date_str);
                } else {
                    println!(
                        "  - No data for {} (likely non-trading day)",
                        trade_date_str
                    );
                }
            }
            Err(e) => {
                eprintln!("  ✗ Error processing {}: {}", trade_date_str, e);
            }
        }

        current_date = current_date + ChronoDuration::days(1);
    }

    println!("\n=== Completed ===");
    println!(
        "Processed {} days from {} to {}",
        processed_days, start_date, end_date
    );
    Ok(())
}

pub async fn pull_insert_by_date(
    dbpool: &sqlx::PgPool,
    trade_date: String,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let client: reqwest::Client = reqwest::Client::new();

    let mut params = HashMap::new();
    params.insert("trade_date".to_string(), trade_date.clone());

    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("moneyflow".to_string(), params))
        .send()
        .await?;

    let response_body = response.text().await?;
    let repx: Responsex = serde_json::from_str(&response_body)?;

    if repx.code != 0 {
        eprintln!("API error for date {}: {}", trade_date, repx.msg);
        return Ok(0);
    }

    let data = repx.data.items;
    if data.is_empty() {
        return Ok(0);
    }

    let mut inserted = 0;

    for item in data {
        let ts_code_val = item[0].as_str().unwrap_or("");
        let trade_date_val = item[1].as_str().unwrap_or("");

        let buy_sm_vol = item[2].as_f64();
        let buy_sm_amount = item[3].as_f64();
        let sell_sm_vol = item[4].as_f64();
        let sell_sm_amount = item[5].as_f64();
        let buy_md_vol = item[6].as_f64();
        let buy_md_amount = item[7].as_f64();
        let sell_md_vol = item[8].as_f64();
        let sell_md_amount = item[9].as_f64();
        let buy_lg_vol = item[10].as_f64();
        let buy_lg_amount = item[11].as_f64();
        let sell_lg_vol = item[12].as_f64();
        let sell_lg_amount = item[13].as_f64();
        let buy_elg_vol = item[14].as_f64();
        let buy_elg_amount = item[15].as_f64();
        let sell_elg_vol = item[16].as_f64();
        let sell_elg_amount = item[17].as_f64();
        let net_mf_vol = item[18].as_f64();
        let net_mf_amount = item[19].as_f64();

        let result = sqlx::query(
            r#"
            INSERT INTO moneyflow (
                trade_date, ts_code,
                buy_sm_vol, buy_sm_amount, sell_sm_vol, sell_sm_amount,
                buy_md_vol, buy_md_amount, sell_md_vol, sell_md_amount,
                buy_lg_vol, buy_lg_amount, sell_lg_vol, sell_lg_amount,
                buy_elg_vol, buy_elg_amount, sell_elg_vol, sell_elg_amount,
                net_mf_vol, net_mf_amount
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            ON CONFLICT (trade_date, ts_code) DO NOTHING
            "#
        )
        .bind(trade_date_val)
        .bind(ts_code_val)
        .bind(buy_sm_vol)
        .bind(buy_sm_amount)
        .bind(sell_sm_vol)
        .bind(sell_sm_amount)
        .bind(buy_md_vol)
        .bind(buy_md_amount)
        .bind(sell_md_vol)
        .bind(sell_md_amount)
        .bind(buy_lg_vol)
        .bind(buy_lg_amount)
        .bind(sell_lg_vol)
        .bind(sell_lg_amount)
        .bind(buy_elg_vol)
        .bind(buy_elg_amount)
        .bind(sell_elg_vol)
        .bind(sell_elg_amount)
        .bind(net_mf_vol)
        .bind(net_mf_amount)
        .execute(dbpool)
        .await;

        match result {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    inserted += 1;
                }
            }
            Err(e) => {
                eprintln!(
                    "Database error for {} on {}: {}",
                    ts_code_val, trade_date_val, e
                );
                continue;
            }
        }
    }

    Ok(inserted)
}
