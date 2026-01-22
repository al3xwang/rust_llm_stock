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

    println!("=== FX Daily Data Pull ===");
    println!("Current date: {}", curr_date.format("%Y-%m-%d"));

    // Define FX pairs to pull (USDCNH = USD/CNH exchange rate)
    let fx_codes = vec![
        "USDCNH.FXCM", // USD/CNH from FXCM
    ];

    println!("Note: Fetching from fx_daily API endpoint");

    // Rate limiter: 200 requests per second
    let mut lim = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(200).unwrap());

    for ts_code in fx_codes.iter() {
        println!("\n=== Processing FX pair: {} ===", ts_code);

        // Check last date for this specific FX pair in index_daily table (reusing for FX data)
        let last_date_for_fx: Option<Option<String>> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM index_daily WHERE ts_code = $1")
                .bind(ts_code)
                .fetch_optional(&pool)
                .await?;

        let fx_start_date = if let Some(Some(ref date_str)) = last_date_for_fx {
            let last_date = NaiveDate::parse_from_str(date_str, "%Y%m%d")?;
            // For FX with existing data, start from the day after the last record
            last_date
                .checked_add_days(Days::new(1))
                .unwrap_or(last_date)
        } else {
            // No data yet: start from 5 years ago
            curr_date.checked_sub_days(Days::new(365 * 5)).unwrap()
        };

        if fx_start_date >= curr_date {
            println!("  {} is already up to date", ts_code);
            continue;
        }

        let mut current_date = fx_start_date;
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

            // Create request for fx_daily data
            let mut params = HashMap::new();
            params.insert("ts_code".to_string(), ts_code.to_string());
            params.insert("start_date".to_string(), start_str.clone());
            params.insert("end_date".to_string(), end_str.clone());

            let request_body = create_req("fx_daily".to_string(), params);

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
                        // Insert into index_daily table (reusing for FX data)
                        for item in items.iter() {
                            if item.len() >= 7 {
                                let ts_code = item[0].as_str();
                                let trade_date = item[1].as_str();
                                // FX data: open, high, low, close (no volume/amount for FX)
                                let open = item[2].as_f64();
                                let high = item[3].as_f64();
                                let low = item[4].as_f64();
                                let close = item[5].as_f64();
                                let pre_close = item[6].as_f64();

                                let insert_result = sqlx::query(
                                    r#"
                                    INSERT INTO index_daily (
                                        ts_code, trade_date, open, high, low, close, pre_close
                                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                                    ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                                        open = EXCLUDED.open,
                                        high = EXCLUDED.high,
                                        low = EXCLUDED.low,
                                        close = EXCLUDED.close,
                                        pre_close = EXCLUDED.pre_close
                                    "#,
                                )
                                .bind(ts_code)
                                .bind(trade_date)
                                .bind(open)
                                .bind(high)
                                .bind(low)
                                .bind(close)
                                .bind(pre_close)
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

    println!("\n=== FX Daily Data Pull Complete ===");

    Ok(())
}

/// Ingest industry money flow data from Tushare (moneyflow_ind_ths)
async fn ingest_industry_moneyflow(
    pool: &sqlx::Pool<sqlx::Postgres>,
    start_date: &str,
    end_date: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rust_llm_stock::stock_db::create_req;
    use std::collections::HashMap;

    println!("=== Ingesting Industry Money Flow Data (moneyflow_ind_ths) ===");
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
        return Ok(());
    }

    let response_body = api_response.text().await?;
    let repx: rust_llm_stock::ts::http::Responsex = match serde_json::from_str(&response_body) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ✗ Failed to parse response: {}", e);
            return Ok(());
        }
    };

    let items = &repx.data.items;
    let mut total_inserted = 0;

    for item in items.iter() {
        // moneyflow_ind_ths fields: ts_code, trade_date, industry_name, buy_sm_vol, buy_sm_amount, sell_sm_vol, sell_sm_amount, buy_md_vol, buy_md_amount, sell_md_vol, sell_md_amount, buy_lg_vol, buy_lg_amount, sell_lg_vol, sell_lg_amount, buy_elg_vol, buy_elg_amount, sell_elg_vol, sell_elg_amount, net_mf_vol, net_mf_amount
        if item.len() >= 20 {
            let ts_code = item[0].as_str();
            let trade_date = item[1].as_str();
            let industry_name = item[2].as_str();
            let net_mf_vol = item[18].as_f64();
            let net_mf_amount = item[19].as_f64();

            let insert_result = sqlx::query(
                r#"
                INSERT INTO moneyflow_industry (
                    ts_code, trade_date, industry_name, net_mf_vol, net_mf_amount
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                    industry_name = EXCLUDED.industry_name,
                    net_mf_vol = EXCLUDED.net_mf_vol,
                    net_mf_amount = EXCLUDED.net_mf_amount
                "#,
            )
            .bind(ts_code)
            .bind(trade_date)
            .bind(industry_name)
            .bind(net_mf_vol)
            .bind(net_mf_amount)
            .execute(pool)
            .await;

            if let Err(e) = insert_result {
                eprintln!("    Error inserting industry moneyflow: {}", e);
            } else {
                total_inserted += 1;
            }
        }
    }

    println!(
        "  ✓ Inserted/updated {} industry moneyflow records",
        total_inserted
    );
    Ok(())
}
