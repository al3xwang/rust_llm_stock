use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Days, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::http::Responsex,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pool = get_connection().await;
    let curr_date = Local::now().date_naive();

    println!("=== Global Index Data Pull ===");
    println!("Current date: {}", curr_date.format("%Y-%m-%d"));

    // Define the global index codes to pull (XIN9 = FTSE China A50, HSI = Hang Seng)
    let index_codes = vec![
        "XIN9", // FTSE China A50
        "HSI",  // Hang Seng Index
    ];

    println!("Note: Fetching from index_global API endpoint");

    // Rate limiter: 200 requests per second
    let mut lim = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(200).unwrap());

    for ts_code in index_codes.iter() {
        println!("\n=== Processing global index: {} ===", ts_code);

        // Check last date for this specific index in index_daily table
        let last_date_for_index: Option<Option<String>> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE ts_code = $1")
                .bind(ts_code)
                .fetch_optional(&pool)
                .await?;

        let index_start_date = if let Some(Some(ref date_str)) = last_date_for_index {
            let last_date = NaiveDate::parse_from_str(date_str, "%Y%m%d")?;
            // For indices with existing data, start from the day after the last record
            last_date
                .checked_add_days(Days::new(1))
                .unwrap_or(last_date)
        } else {
            // No data yet: start from 5 years ago
            curr_date.checked_sub_days(Days::new(365 * 5)).unwrap()
        };

        if index_start_date >= curr_date {
            println!("  {} is already up to date", ts_code);
            continue;
        }

        let mut current_date = index_start_date;
        let mut total_records = 0;

        while current_date < curr_date {
            // Calculate end date for this chunk (30 days at a time to avoid timeouts)
            let end_date = (current_date.checked_add_days(Days::new(30)).unwrap()).min(curr_date);

            let start_str = current_date.format("%Y%m%d").to_string();
            let end_str = end_date.format("%Y%m%d").to_string();

            println!("  Fetching {} from {} to {}", ts_code, start_str, end_str);

            // Wait for rate limiter
            while lim.check().is_err() {
                thread::sleep(Duration::from_millis(100));
            }

            // Create request for index_global data (different API endpoint)
            let mut params = HashMap::new();
            params.insert("ts_code".to_string(), ts_code.to_string());
            params.insert("start_date".to_string(), start_str.clone());
            params.insert("end_date".to_string(), end_str.clone());

            let request_body = create_req("index_global".to_string(), params);

            // Make HTTP request to API
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

            // Parse response
            match serde_json::from_str::<Responsex>(&response_body) {
                Ok(repx) => {
                    let items = &repx.data.items;
                    let record_count = items.len();

                    if record_count > 0 {
                        // Insert into index_daily table (same table as domestic indices)
                        for item in items.iter() {
                            if item.len() >= 11 {
                                let ts_code = item[0].as_str();
                                let trade_date = item[1].as_str();
                                let open = item[2].as_f64();
                                let high = item[3].as_f64();
                                let low = item[4].as_f64();
                                let close = item[5].as_f64();
                                let pre_close = item[6].as_f64();
                                let change = item[7].as_f64();
                                let pct_chg = item[8].as_f64();
                                let vol = item[9].as_f64();
                                let amount = item[10].as_f64();

                                let insert_result = sqlx::query(
                                    r#"
                                    INSERT INTO index_daily (
                                        ts_code, trade_date, open, high, low, close, 
                                        pre_close, change, pct_chg, vol, amount
                                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                                    ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                                        open = EXCLUDED.open,
                                        high = EXCLUDED.high,
                                        low = EXCLUDED.low,
                                        close = EXCLUDED.close,
                                        pre_close = EXCLUDED.pre_close,
                                        change = EXCLUDED.change,
                                        pct_chg = EXCLUDED.pct_chg,
                                        vol = EXCLUDED.vol,
                                        amount = EXCLUDED.amount
                                    "#,
                                )
                                .bind(ts_code)
                                .bind(trade_date)
                                .bind(open)
                                .bind(high)
                                .bind(low)
                                .bind(close)
                                .bind(pre_close)
                                .bind(change)
                                .bind(pct_chg)
                                .bind(vol)
                                .bind(amount)
                                .execute(&pool)
                                .await;

                                if let Err(e) = insert_result {
                                    eprintln!("    Error inserting record: {}", e);
                                }
                            }
                        }

                        total_records += record_count;
                        println!("    ✓ Inserted {} records", record_count);
                    } else {
                        println!("    - No data for this period");
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ Failed to parse response: {}", e);
                }
            }

            // Move to next chunk
            current_date = end_date.checked_add_days(Days::new(1)).unwrap();

            // Small delay between requests
            thread::sleep(Duration::from_millis(300));
        }

        println!(
            "  ✓ {} complete: {} total records inserted",
            ts_code, total_records
        );
    }

    println!("\n=== Global Index Data Pull Complete ===");

    Ok(())
}
