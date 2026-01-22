use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Duration as ChronoDuration, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::http::Responsex,
};

#[tokio::main]
async fn main() {
    // Load environment variables from .env file
    dotenv::dotenv().ok();

    match run().await {
        Ok(_) => println!("Success"),
        Err(e) => {
            eprintln!("Fatal error: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Tushare rate limit: keep modest to avoid 429
    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());

    println!("Connecting to database...");
    let dbpool = get_connection().await;
    println!("Connected to database");

    // Query existing max(trade_date) for backward iteration
    let stop_date: Option<NaiveDate> = match sqlx::query_scalar::<_, Option<String>>(
        "SELECT MAX(trade_date) FROM moneyflow_ind_dc WHERE trade_date ~ '^[0-9]{8}$'",
    )
    .fetch_optional(&dbpool)
    .await
    {
        Ok(Some(Some(date_str))) => {
            if let Ok(date) = NaiveDate::parse_from_str(&date_str, "%Y%m%d") {
                println!("  Existing data up to: {}", date.format("%Y-%m-%d"));
                Some(date)
            } else {
                None
            }
        }
        Ok(_) => {
            println!("  No existing data in table");
            None
        }
        Err(e) => {
            eprintln!("Database query error: {}", e);
            return Err(format!("Failed to fetch max trade_date: {}", e).into());
        }
    };

    let curr_date = Local::now().naive_local().date();
    println!(
        "  Starting backward ingestion from: {}",
        curr_date.format("%Y-%m-%d")
    );

    println!("\n=== Processing Industry Moneyflow Data (moneyflow_ind_dc API - Backward) ===\n");

    let mut current_date = curr_date;
    let mut total_inserted = 0;
    let mut consecutive_empty_days = 0;
    const MAX_EMPTY_DAYS: i32 = 15;

    loop {
        // Stopping condition 1: Reached existing data
        if let Some(ref max_date) = stop_date {
            if current_date <= *max_date {
                println!("✓ Reached existing data at {}", max_date.format("%Y-%m-%d"));
                break;
            }
        }

        // Stopping condition 2: 15 consecutive empty days
        if consecutive_empty_days >= MAX_EMPTY_DAYS {
            println!(
                "✓ Early stopping: {} consecutive days with no data",
                consecutive_empty_days
            );
            break;
        }

        // Rate limiting
        while limiter.check().is_err() {
            thread::sleep(Duration::from_millis(100));
        }

        let trade_date = current_date.format("%Y%m%d").to_string();

        match pull_one_day(&dbpool, &trade_date).await {
            Ok(inserted) => {
                if inserted > 0 {
                    consecutive_empty_days = 0; // Reset counter on success
                    total_inserted += inserted;
                    print!(".");
                } else {
                    consecutive_empty_days += 1; // Increment on empty
                    print!("·");
                }
            }
            Err(e) => {
                consecutive_empty_days += 1; // Increment on error
                eprintln!("\n  ✗ error for {}: {}", trade_date, e);
                print!("·");
            }
        }

        // Move backward one day
        current_date = current_date
            .checked_sub_signed(ChronoDuration::days(1))
            .unwrap();

        thread::sleep(Duration::from_millis(300));
    }

    println!("\n");
    println!("Done. Inserted {} total rows.", total_inserted);
    Ok(())
}

async fn pull_one_day(
    dbpool: &sqlx::PgPool,
    trade_date: &str,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();

    let mut params = HashMap::new();
    params.insert("trade_date".to_string(), trade_date.to_string());

    let resp = client
        .post("http://api.tushare.pro")
        .body(create_req("moneyflow_ind_dc".to_string(), params))
        .send()
        .await?;

    let text = resp.text().await?;
    let repx: Responsex = match serde_json::from_str(&text) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("JSON parse error for {}: {}", trade_date, e);
            eprintln!("Response text: {}", &text);
            return Err(format!("JSON parse failed: {}", e).into());
        }
    };

    if repx.code != 0 {
        eprintln!("API error {}: {}", trade_date, repx.msg);
        return Ok(0);
    }

    let items = repx.data.items;
    if items.is_empty() {
        return Ok(0);
    }

    let mut inserted = 0usize;

    for row in items {
        // Expected order: ts_code, trade_date, industry_name, net_buy_amount, net_sell_amount, net_amount
        let ts_code = row.get(0).and_then(|v| v.as_str()).unwrap_or("");
        let trade_date_val = row.get(1).and_then(|v| v.as_str()).unwrap_or("");
        let industry_name = row.get(2).and_then(|v| v.as_str());
        let net_buy_amount = row.get(3).and_then(|v| v.as_f64());
        let net_sell_amount = row.get(4).and_then(|v| v.as_f64());
        let net_amount = row.get(5).and_then(|v| v.as_f64());

        let res = sqlx::query(
            r#"
            INSERT INTO moneyflow_ind_dc (
                ts_code, trade_date, industry_name,
                net_buy_amount, net_sell_amount, net_amount
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (ts_code, trade_date) DO NOTHING
            "#,
        )
        .bind(ts_code)
        .bind(trade_date_val)
        .bind(industry_name)
        .bind(net_buy_amount)
        .bind(net_sell_amount)
        .bind(net_amount)
        .execute(dbpool)
        .await;

        match res {
            Ok(r) => {
                if r.rows_affected() > 0 {
                    inserted += 1;
                }
            }
            Err(e) => {
                eprintln!("DB error for {} {}: {}", ts_code, trade_date_val, e);
            }
        }
    }

    Ok(inserted)
}
