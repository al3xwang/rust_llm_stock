use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Days, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::http::Responsex,
};
use sqlx::Postgres;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let pool = get_connection().await;
    ensure_table(&pool).await?;

    let curr_date = Local::now().date_naive();
    println!("=== Macro Daily Data Pull (USD/CNY FX and Gold) ===");
    println!("Current date: {}", curr_date.format("%Y-%m-%d"));

    // Rate limiter: 200 requests per second
    let mut lim = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(200).unwrap());

    // Ingest USD/CNY FX daily
    println!("\n=== Fetching USD/CNY Exchange Rate ===");
    ingest_fx(&pool, &mut lim, curr_date).await?;

    // Ingest Gold prices daily
    println!("\n=== Fetching Gold Price ===");
    ingest_gold(&pool, &mut lim, curr_date).await?;

    println!("\n=== Macro Data Pull Complete ===");
    Ok(())
}

use std::error::Error;

async fn ensure_table(pool: &sqlx::Pool<Postgres>) -> Result<(), Box<dyn Error + Send + Sync>> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS macro_daily (
            trade_date VARCHAR(10) PRIMARY KEY,
            usd_cny_rate DOUBLE PRECISION,
            usd_cny_pct_chg DOUBLE PRECISION,
            gold_price DOUBLE PRECISION,
            gold_pct_chg DOUBLE PRECISION,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        "#,
    )
    .execute(pool)
    .await?;

    println!("✓ macro_daily table ready");
    Ok(())
}

async fn ingest_fx(
    pool: &sqlx::Pool<Postgres>,
    lim: &mut DirectRateLimiter<GCRA>,
    curr_date: NaiveDate,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Fetch USD/CNY FX data
    // "forex_daily" endpoint with ts_code = "USDCNY"
    // Get max date from macro_daily to determine start point
    let max_date_result: Option<Option<String>> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM macro_daily")
            .fetch_optional(pool)
            .await?;

    let start_date = if let Some(Some(date_str)) = max_date_result {
        let last_date = NaiveDate::parse_from_str(&date_str, "%Y%m%d")?;
        last_date
            .checked_add_days(Days::new(1))
            .unwrap_or(last_date)
    } else {
        // Default: start from 5 years ago
        curr_date.checked_sub_days(Days::new(365 * 5)).unwrap()
    };

    if start_date >= curr_date {
        println!("USD/CNY FX data is already up to date!");
        return Ok(());
    }

    println!(
        "Pulling USD/CNY FX from {} to {}",
        start_date.format("%Y%m%d"),
        curr_date.format("%Y%m%d")
    );

    let mut current_date = start_date;
    let mut total_records = 0;

    while current_date < curr_date {
        let end_date = (current_date.checked_add_days(Days::new(30)).unwrap()).min(curr_date);
        let start_str = current_date.format("%Y%m%d").to_string();
        let end_str = end_date.format("%Y%m%d").to_string();

        println!("  Fetching USD/CNY from {} to {}", start_str, end_str);

        while lim.check().is_err() {
            thread::sleep(Duration::from_millis(100));
        }

        let mut params = HashMap::new();
        params.insert("ts_code".to_string(), "USDCNY".to_string());
        params.insert("start_date".to_string(), start_str.clone());
        params.insert("end_date".to_string(), end_str.clone());

        let request_body = create_req("forex_daily".to_string(), params);

        let client = reqwest::Client::new();
        let api_response = client
            .post("http://api.tushare.pro")
            .body(request_body)
            .send()
            .await;

        let response_body = match api_response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    eprintln!("    ✗ API request failed with status: {}", resp.status());
                    current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                    thread::sleep(Duration::from_millis(300));
                    continue;
                }
                match resp.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        eprintln!("    ✗ Failed to read response: {}", e);
                        current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                        thread::sleep(Duration::from_millis(300));
                        continue;
                    }
                }
            }
            Err(e) => {
                eprintln!("    ✗ HTTP request failed: {}", e);
                current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                thread::sleep(Duration::from_millis(300));
                continue;
            }
        };

        match serde_json::from_str::<Responsex>(&response_body) {
            Ok(repx) => {
                let items = &repx.data.items;
                let record_count = items.len();

                if record_count > 0 {
                    for item in items.iter() {
                        if item.len() >= 5 {
                            let trade_date = item[0].as_str();
                            let close = item[3].as_f64(); // Close rate
                            let pct_chg = item[4].as_f64(); // % change

                            let insert_result = sqlx::query(
                                r#"
                                INSERT INTO macro_daily (trade_date, usd_cny_rate, usd_cny_pct_chg)
                                VALUES ($1, $2, $3)
                                ON CONFLICT (trade_date) DO UPDATE SET
                                    usd_cny_rate = COALESCE(EXCLUDED.usd_cny_rate, macro_daily.usd_cny_rate),
                                    usd_cny_pct_chg = COALESCE(EXCLUDED.usd_cny_pct_chg, macro_daily.usd_cny_pct_chg)
                                "#,
                            )
                            .bind(trade_date)
                            .bind(close)
                            .bind(pct_chg)
                            .execute(pool)
                            .await;

                            if let Err(e) = insert_result {
                                eprintln!("    Error inserting FX record: {}", e);
                            }
                        }
                    }
                    total_records += record_count;
                    println!("    ✓ Inserted {} FX records", record_count);
                } else {
                    println!("    - No FX data for this period");
                }
            }
            Err(e) => {
                eprintln!("    ✗ Failed to parse response: {}", e);
            }
        }

        current_date = end_date.checked_add_days(Days::new(1)).unwrap();
        thread::sleep(Duration::from_millis(300));
    }

    println!(
        "  ✓ USD/CNY complete: {} total records inserted",
        total_records
    );
    Ok(())
}

async fn ingest_gold(
    pool: &sqlx::Pool<Postgres>,
    lim: &mut DirectRateLimiter<GCRA>,
    curr_date: NaiveDate,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Fetch Gold prices (Tushare futures data for AU9999 - Shanghai Gold Exchange)
    // Gold closing price and pct_chg from "futures_daily"
    // Note: May also use commodity_daily or futures_daily depending on Tushare availability

    let max_date_result: Option<Option<String>> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM macro_daily")
            .fetch_optional(pool)
            .await?;

    let start_date = if let Some(Some(date_str)) = max_date_result {
        let last_date = NaiveDate::parse_from_str(&date_str, "%Y%m%d")?;
        last_date
            .checked_add_days(Days::new(1))
            .unwrap_or(last_date)
    } else {
        curr_date.checked_sub_days(Days::new(365 * 5)).unwrap()
    };

    if start_date >= curr_date {
        println!("Gold price data is already up to date!");
        return Ok(());
    }

    println!(
        "Pulling Gold (AU9999) from {} to {}",
        start_date.format("%Y%m%d"),
        curr_date.format("%Y%m%d")
    );

    let mut current_date = start_date;
    let mut total_records = 0;

    while current_date < curr_date {
        let end_date = (current_date.checked_add_days(Days::new(30)).unwrap()).min(curr_date);
        let start_str = current_date.format("%Y%m%d").to_string();
        let end_str = end_date.format("%Y%m%d").to_string();

        println!("  Fetching Gold (AU9999) from {} to {}", start_str, end_str);

        while lim.check().is_err() {
            thread::sleep(Duration::from_millis(100));
        }

        let mut params = HashMap::new();
        params.insert("ts_code".to_string(), "AU9999.SGE".to_string()); // Shanghai Gold Exchange
        params.insert("start_date".to_string(), start_str.clone());
        params.insert("end_date".to_string(), end_str.clone());

        let request_body = create_req("futures_daily".to_string(), params);

        let client = reqwest::Client::new();
        let api_response = client
            .post("http://api.tushare.pro")
            .body(request_body)
            .send()
            .await;

        let response_body = match api_response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    eprintln!("    ✗ API request failed with status: {}", resp.status());
                    current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                    thread::sleep(Duration::from_millis(300));
                    continue;
                }
                match resp.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        eprintln!("    ✗ Failed to read response: {}", e);
                        current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                        thread::sleep(Duration::from_millis(300));
                        continue;
                    }
                }
            }
            Err(e) => {
                eprintln!("    ✗ HTTP request failed: {}", e);
                current_date = end_date.checked_add_days(Days::new(1)).unwrap();
                thread::sleep(Duration::from_millis(300));
                continue;
            }
        };

        match serde_json::from_str::<Responsex>(&response_body) {
            Ok(repx) => {
                let items = &repx.data.items;
                let record_count = items.len();

                if record_count > 0 {
                    for item in items.iter() {
                        if item.len() >= 5 {
                            let trade_date = item[1].as_str(); // trade_date is at index 1
                            let close = item[4].as_f64(); // Close price
                            let pct_chg = item[8].as_f64(); // % change (index 8)

                            let insert_result = sqlx::query(
                                r#"
                                INSERT INTO macro_daily (trade_date, gold_price, gold_pct_chg)
                                VALUES ($1, $2, $3)
                                ON CONFLICT (trade_date) DO UPDATE SET
                                    gold_price = COALESCE(EXCLUDED.gold_price, macro_daily.gold_price),
                                    gold_pct_chg = COALESCE(EXCLUDED.gold_pct_chg, macro_daily.gold_pct_chg)
                                "#,
                            )
                            .bind(trade_date)
                            .bind(close)
                            .bind(pct_chg)
                            .execute(pool)
                            .await;

                            if let Err(e) = insert_result {
                                eprintln!("    Error inserting gold record: {}", e);
                            }
                        }
                    }
                    total_records += record_count;
                    println!("    ✓ Inserted {} gold records", record_count);
                } else {
                    println!("    - No gold data for this period");
                }
            }
            Err(e) => {
                eprintln!("    ✗ Failed to parse response: {}", e);
            }
        }

        current_date = end_date.checked_add_days(Days::new(1)).unwrap();
        thread::sleep(Duration::from_millis(300));
    }

    println!(
        "  ✓ Gold complete: {} total records inserted",
        total_records
    );
    Ok(())
}
