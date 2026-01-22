use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Days, Local, NaiveDate};
use ratelimit_meter::{DirectRateLimiter, GCRA};
use rust_llm_stock::{
    stock_db::{create_req, get_connection},
    ts::http::Responsex,
};
use sqlx::Postgres;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pool = get_connection().await;
    let curr_date = Local::now().date_naive();

    println!("=== Index Daily Data Pull ===");
    println!("Current date: {}", curr_date.format("%Y-%m-%d"));

    // Define the index codes to pull (domestic indices - use index_daily API)
    let index_codes = vec![
        "000300.SH",  // CSI 300
        "399006.SZ",  // ChiNext Index
        "000001.SH",  // Shanghai Composite (上证指数)
        "399001.SZ",  // Shenzhen Component (深证成指)
        "932000.CSI", // CSI 1000 (Mid-cap)
        "830002.XI",  // Additional index (user requested)
    ];

    // Define global index codes (use index_global API)
    let global_index_codes = vec![
        "XIN9", // XIN9 Index (used in ml_training_dataset)
        "HSI",  // Hang Seng Index
    ];

    // Define forex codes (use fx_daily API)
    let forex_codes = vec![
        "USDCNH.FXCM", // USD/CNH exchange rate
    ];

    // Optional: global max date (informational)
    let global_max_date: Option<String> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily")
            .fetch_optional(&pool)
            .await?
            .flatten();

    if let Some(ref date_str) = global_max_date {
        if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y%m%d") {
            println!(
                "Global max trade_date in index_daily: {}",
                date.format("%Y-%m-%d")
            );
        }
    } else {
        println!("No data found in index_daily yet; will use base_date per index");
    }

    // Rate limiter: 200 requests per second
    let mut lim = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(200).unwrap());

    // Process domestic indices first
    println!("=== Processing Domestic Indices (index_daily API) ===\n");
    for ts_code in index_codes.iter() {
        println!("\n=== Processing index: {} ===", ts_code);

        // Get max date in table (backward iteration stopping point)
        let table_max_date: Option<String> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE ts_code = $1")
                .bind(ts_code)
                .fetch_optional(&pool)
                .await?
                .flatten();

        let stop_date = if let Some(ref date_str) = table_max_date {
            match NaiveDate::parse_from_str(date_str, "%Y%m%d") {
                Ok(date) => Some(date),
                Err(_) => {
                    eprintln!("  ⚠ Invalid max date format for {}: {}", ts_code, date_str);
                    None
                }
            }
        } else {
            None
        };

        if let Some(ref max_date) = stop_date {
            println!("  Existing data up to: {}", max_date.format("%Y-%m-%d"));
        } else {
            println!("  No existing data in table");
        }

        println!(
            "  Starting backward ingestion from: {}",
            curr_date.format("%Y-%m-%d")
        );

        let mut current_date = curr_date;
        let mut total_records = 0;
        let mut consecutive_empty_days = 0;
        const MAX_EMPTY_DAYS: i32 = 15;

        loop {
            // Check if we've reached the existing data
            if let Some(ref max_date) = stop_date {
                if current_date <= *max_date {
                    println!(
                        "  ✓ Reached existing data at {}",
                        max_date.format("%Y-%m-%d")
                    );
                    break;
                }
            }

            // Check early stopping (15 consecutive empty days)
            if consecutive_empty_days >= MAX_EMPTY_DAYS {
                println!(
                    "  ✓ Early stopping: {} consecutive days with no data",
                    consecutive_empty_days
                );
                break;
            }

            // Fetch this single day
            let start_str = current_date.format("%Y%m%d").to_string();
            let end_str = start_str.clone();

            // Wait for rate limiter
            while lim.check().is_err() {
                thread::sleep(Duration::from_millis(100));
            }

            // Create request for index daily data
            let mut params = HashMap::new();
            params.insert("ts_code".to_string(), ts_code.to_string());
            params.insert("start_date".to_string(), start_str.clone());
            params.insert("end_date".to_string(), end_str.clone());

            let request_body = create_req("index_daily".to_string(), params);

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
                        eprintln!(
                            "    ✗ API request failed ({}) for {}",
                            resp.status(),
                            start_str
                        );
                        consecutive_empty_days += 1;
                        current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                        thread::sleep(Duration::from_millis(300));
                        continue;
                    }
                    match resp.text().await {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!("    ✗ Failed to read response: {}", e);
                            consecutive_empty_days += 1;
                            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                            thread::sleep(Duration::from_millis(300));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ HTTP request failed: {}", e);
                    consecutive_empty_days += 1;
                    current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
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
                        consecutive_empty_days = 0; // Reset counter on success

                        // Insert into index_daily table
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
                        print!(".");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    } else {
                        consecutive_empty_days += 1;
                        print!("·");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ Failed to parse response: {}", e);
                    consecutive_empty_days += 1;
                }
            }

            // Move backward one day
            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
            thread::sleep(Duration::from_millis(300));
        }

        println!();
        println!(
            "  ✓ {} complete: {} total records inserted",
            ts_code, total_records
        );
    }

    // Process global indices (using index_global API) - BACKWARD ITERATION
    println!("\n=== Processing Global Indices (index_global API - Backward) ===\n");
    for ts_code in global_index_codes.iter() {
        println!("\n=== Global Index (Backward): {} ===", ts_code);

        // Query existing max date as stopping point
        let table_max_date: Option<String> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE ts_code = $1")
                .bind(ts_code)
                .fetch_optional(&pool)
                .await?
                .flatten();

        let stop_date = if let Some(ref date_str) = table_max_date {
            match NaiveDate::parse_from_str(date_str, "%Y%m%d") {
                Ok(date) => Some(date),
                Err(_) => {
                    eprintln!("  ✗ Invalid max date format: {}", date_str);
                    None
                }
            }
        } else {
            None
        };

        if let Some(ref max_date) = stop_date {
            println!(
                "  Existing data up to: {} (starting backward from today)",
                max_date.format("%Y-%m-%d")
            );
        } else {
            println!("  No existing data (starting backward from today)");
        }

        let mut current_date = curr_date;
        let mut consecutive_empty_days = 0;
        const MAX_EMPTY_DAYS: i32 = 15;
        let mut total_records = 0;

        loop {
            // Check stopping conditions
            if let Some(ref max_date) = stop_date {
                if current_date <= *max_date {
                    println!(
                        "  ✓ Reached existing data at {}",
                        max_date.format("%Y-%m-%d")
                    );
                    break;
                }
            }

            if consecutive_empty_days >= MAX_EMPTY_DAYS {
                println!(
                    "  ✓ Early stopping: {} consecutive days with no data",
                    consecutive_empty_days
                );
                break;
            }

            // Single day fetching (backward iteration)
            let start_str = current_date.format("%Y%m%d").to_string();
            let end_str = start_str.clone();

            // Rate limiting
            while lim.check().is_err() {
                thread::sleep(Duration::from_millis(100));
            }

            // Use index_global API for global indices
            let mut params = HashMap::new();
            params.insert("ts_code".to_string(), ts_code.to_string());
            params.insert("start_date".to_string(), start_str.clone());
            params.insert("end_date".to_string(), end_str.clone());

            let request_body = create_req("index_global".to_string(), params);

            let client = reqwest::Client::new();
            let api_response = client
                .post("http://api.tushare.pro")
                .body(request_body)
                .send()
                .await;

            let response_body = match api_response {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        eprintln!(
                            "    ✗ API request failed ({}) for {}",
                            resp.status(),
                            start_str
                        );
                        consecutive_empty_days += 1;
                        current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                        thread::sleep(Duration::from_millis(300));
                        continue;
                    }
                    match resp.text().await {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!("    ✗ Failed to read response: {}", e);
                            consecutive_empty_days += 1;
                            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                            thread::sleep(Duration::from_millis(300));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ HTTP request failed: {}", e);
                    consecutive_empty_days += 1;
                    current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
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
                        consecutive_empty_days = 0; // Reset counter on success

                        // Insert into index_daily table
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
                        print!(".");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    } else {
                        consecutive_empty_days += 1;
                        print!("·");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ Failed to parse response: {}", e);
                    consecutive_empty_days += 1;
                }
            }

            // Move backward one day
            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
            thread::sleep(Duration::from_millis(300));
        }

        println!();
        println!(
            "  ✓ {} complete: {} total records inserted",
            ts_code, total_records
        );
    }

    // Process forex pairs (using fx_daily API) - BACKWARD ITERATION
    println!("\n=== Processing Forex Pairs (fx_daily API - Backward) ===\n");
    for ts_code in forex_codes.iter() {
        println!("\n=== Forex (Backward): {} ===", ts_code);

        // Query existing max date as stopping point
        let table_max_date: Option<String> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE ts_code = $1")
                .bind(ts_code)
                .fetch_optional(&pool)
                .await?
                .flatten();

        let stop_date = if let Some(ref date_str) = table_max_date {
            match NaiveDate::parse_from_str(date_str, "%Y%m%d") {
                Ok(date) => Some(date),
                Err(_) => {
                    eprintln!("  ✗ Invalid max date format: {}", date_str);
                    None
                }
            }
        } else {
            None
        };

        if let Some(ref max_date) = stop_date {
            println!(
                "  Existing data up to: {} (starting backward from today)",
                max_date.format("%Y-%m-%d")
            );
        } else {
            println!("  No existing data (starting backward from today)");
        }

        let mut current_date = curr_date;
        let mut consecutive_empty_days = 0;
        const MAX_EMPTY_DAYS: i32 = 15;
        let mut total_records = 0;

        loop {
            // Check stopping conditions
            if let Some(ref max_date) = stop_date {
                if current_date <= *max_date {
                    println!(
                        "  ✓ Reached existing data at {}",
                        max_date.format("%Y-%m-%d")
                    );
                    break;
                }
            }

            if consecutive_empty_days >= MAX_EMPTY_DAYS {
                println!(
                    "  ✓ Early stopping: {} consecutive days with no data",
                    consecutive_empty_days
                );
                break;
            }

            // Single day fetching (backward iteration)
            let start_str = current_date.format("%Y%m%d").to_string();
            let end_str = start_str.clone();

            // Rate limiting
            while lim.check().is_err() {
                thread::sleep(Duration::from_millis(100));
            }

            // Use fx_daily API for forex pairs
            let mut params = HashMap::new();
            params.insert("ts_code".to_string(), ts_code.to_string());
            params.insert("start_date".to_string(), start_str.clone());
            params.insert("end_date".to_string(), end_str.clone());

            let request_body = create_req("fx_daily".to_string(), params);

            let client = reqwest::Client::new();
            let api_response = client
                .post("http://api.tushare.pro")
                .body(request_body)
                .send()
                .await;

            let response_body = match api_response {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        eprintln!(
                            "    ✗ API request failed ({}) for {}",
                            resp.status(),
                            start_str
                        );
                        consecutive_empty_days += 1;
                        current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                        thread::sleep(Duration::from_millis(300));
                        continue;
                    }
                    match resp.text().await {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!("    ✗ Failed to read response: {}", e);
                            consecutive_empty_days += 1;
                            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
                            thread::sleep(Duration::from_millis(300));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ HTTP request failed: {}", e);
                    consecutive_empty_days += 1;
                    current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
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
                        consecutive_empty_days = 0; // Reset counter on success

                        // Insert into index_daily table (same table for all index-like data)
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
                                    "INSERT INTO index_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                                     ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                                     open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                                     close = EXCLUDED.close, pre_close = EXCLUDED.pre_close,
                                     change = EXCLUDED.change, pct_chg = EXCLUDED.pct_chg,
                                     vol = EXCLUDED.vol, amount = EXCLUDED.amount"
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
                        print!(".");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    } else {
                        consecutive_empty_days += 1;
                        print!("·");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }
                Err(e) => {
                    eprintln!("    ✗ Failed to parse response: {}", e);
                    consecutive_empty_days += 1;
                }
            }

            // Move backward one day
            current_date = current_date.checked_sub_days(Days::new(1)).unwrap();
            thread::sleep(Duration::from_millis(300));
        }

        println!();
        println!(
            "  ✓ {} complete: {} total records inserted",
            ts_code, total_records
        );
    }

    println!("\n=== Index Data Pull Complete ===");

    Ok(())
}
