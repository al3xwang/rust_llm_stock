use std::{collections::HashMap, num::NonZeroU32, thread, time::Duration};

use chrono::{Datelike, Days, Local, NaiveDate, Weekday};
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

    println!("=== Daily Data Pull - Per-Day All-Stocks Update ===");
    println!("Current date: {}", curr_date.format("%Y-%m-%d"));

    // Get the max trade_date from stock_daily table (incremental update)
    let max_trade_date: Option<String> =
        sqlx::query_scalar("SELECT MAX(trade_date) FROM stock_daily")
            .fetch_optional(&pool)
            .await?;

    let start_date = if let Some(max_date_str) = max_trade_date {
        // Start from the day after max_trade_date
        let max_date = NaiveDate::parse_from_str(&max_date_str, "%Y%m%d")?;
        let next_date = max_date.checked_add_days(Days::new(1)).unwrap_or(max_date);
        println!(
            "Found existing data up to {}, starting from {}",
            max_date.format("%Y-%m-%d"),
            next_date.format("%Y-%m-%d")
        );
        next_date
    } else {
        // No existing data, get the earliest list_date from stock_basic
        let earliest_list_date: Option<String> = sqlx::query_scalar(
            "SELECT MIN(list_date) FROM stock_basic WHERE list_date IS NOT NULL AND list_date != ''",
        )
        .fetch_optional(&pool)
        .await?;

        let date = if let Some(date_str) = earliest_list_date {
            NaiveDate::parse_from_str(&date_str, "%Y%m%d")?
        } else {
            NaiveDate::from_ymd_opt(1990, 1, 1).unwrap()
        };
        println!(
            "No existing data found, starting from {}",
            date.format("%Y-%m-%d")
        );
        date
    };

    // Fetch all trading days (no params)
    let trading_days = fetch_all_trading_days().await?;
    println!("Fetched {} trading days from API", trading_days.len());

    // Filter trading days to only those between start_date and curr_date
    let trading_days: Vec<String> = trading_days
        .into_iter()
        .filter(|d| {
            if let Ok(date) = NaiveDate::parse_from_str(d, "%Y%m%d") {
                date >= start_date && date <= curr_date
            } else {
                false
            }
        })
        .collect();

    println!(
        "Processing {} trading days (from {} to {})",
        trading_days.len(),
        start_date.format("%Y-%m-%d"),
        curr_date.format("%Y-%m-%d")
    );

    let mut total_records = 0;
    let mut limiter = DirectRateLimiter::<GCRA>::per_second(NonZeroU32::new(3).unwrap());
    for (idx, date_str) in trading_days.iter().enumerate() {
        // Wait for rate limit
        loop {
            if limiter.check().is_ok() {
                break;
            } else {
                thread::sleep(Duration::from_millis(100));
            }
        }
        // Pull all stocks' data for this day
        let mut params = HashMap::new();
        params.insert("trade_date".to_string(), date_str.clone());
        let client = reqwest::Client::new();
        let response = client
            .post("http://api.tushare.pro")
            .body(create_req("daily".to_string(), params))
            .send()
            .await?;
        if !response.status().is_success() {
            eprintln!(
                "✗ API request failed for {}: {}",
                date_str,
                response.status()
            );
            continue;
        }
        let response_body = response.text().await?;
        let repx: Responsex = serde_json::from_str(&response_body)?;
        let mut records_inserted = 0;
        for item in &repx.data.items {
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
            let result = sqlx::query(
                "INSERT INTO stock_daily (ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (ts_code, trade_date) DO NOTHING"
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
            .await?;
            if result.rows_affected() > 0 {
                records_inserted += 1;
            }
        }
        total_records += records_inserted;
        println!(
            "[{}] {} - {} records inserted",
            idx + 1,
            date_str,
            records_inserted
        );
    }
    println!("\n=== Pull Complete ===");
    println!("✓ Successfully inserted {} records", total_records);
    Ok(())
}

/// Fetch all trading days (no params)
async fn fetch_all_trading_days() -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let params = HashMap::new(); // No params: get all trading days
    let response = client
        .post("http://api.tushare.pro")
        .body(create_req("trade_cal".to_string(), params))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("Trade calendar API failed: {}", response.status()).into());
    }

    let response_body = response.text().await?;
    let repx: Responsex = serde_json::from_str(&response_body)?;
    let mut trading_days = Vec::new();
    for item in &repx.data.items {
        let cal_date = item[1].as_str().unwrap_or("");
        let is_open = if let Some(num) = item[2].as_u64() {
            num == 1
        } else if let Some(s) = item[2].as_str() {
            s == "1"
        } else {
            false
        };
        if is_open {
            trading_days.push(cal_date.to_string());
        }
    }
    Ok(trading_days)
}

// SQL queries for aggregating industry and market data:

// 处理行业板块日交易数据
// INSERT INTO public.industry (ts_code, trade_date, open, high, low, close, vol, amount, change, pct_chg, pre_close)
// SELECT
//     industry as ts_code,
//     trade_date,
//     ROUND(AVG(open)::numeric, 2) as open,
//     ROUND(AVG(high)::numeric, 2) AS high,
//     ROUND(AVG(low)::numeric, 2) AS low,
//     ROUND(AVG(close)::numeric, 2) AS close,
//     ROUND(SUM(vol)::numeric, 2) AS vol,
//     ROUND(SUM(amount)::numeric, 2) as amount,
//     ROUND(AVG(change)::numeric, 2) as change,
//     ROUND(AVG(pct_chg)::numeric, 2) as pct_chg,
//     ROUND(AVG(pre_close)::numeric, 2) as pre_close
// FROM stock_basic b
// JOIN daily d ON b.ts_code = d.ts_code
// WHERE name NOT LIKE 'ST%'
// GROUP BY trade_date, industry
// ORDER BY trade_date
// ON CONFLICT (ts_code, trade_date) DO UPDATE SET
//     open = EXCLUDED.open,
//     high = EXCLUDED.high,
//     low = EXCLUDED.low,
//     close = EXCLUDED.close,
//     vol = EXCLUDED.vol,
//     amount = EXCLUDED.amount,
//     change = EXCLUDED.change,
//     pct_chg = EXCLUDED.pct_chg,
//     pre_close = EXCLUDED.pre_close;

// 处理市场板块日交易数据
// INSERT INTO public.market (ts_code, trade_date, open, high, low, close, vol, amount, change, pct_chg, pre_close)
// SELECT
//     SUBSTR(b.ts_code, 1, 3) as ts_code,
//     trade_date,
//     ROUND(AVG(open)::numeric, 2) as open,
//     ROUND(AVG(high)::numeric, 2) AS high,
//     ROUND(AVG(low)::numeric, 2) AS low,
//     ROUND(AVG(close)::numeric, 2) AS close,
//     ROUND(SUM(vol)::numeric, 2) AS vol,
//     ROUND(SUM(amount)::numeric, 2) as amount,
//     ROUND(AVG(change)::numeric, 2) as change,
//     ROUND(AVG(pct_chg)::numeric, 2) as pct_chg,
//     ROUND(AVG(pre_close)::numeric, 2) as pre_close
// FROM stock_basic b
// JOIN daily d ON b.ts_code = d.ts_code
// WHERE name NOT LIKE 'ST%'
// GROUP BY SUBSTR(b.ts_code, 1, 3), trade_date
// ORDER BY trade_date
// ON CONFLICT (ts_code, trade_date) DO UPDATE SET
//     open = EXCLUDED.open,
//     high = EXCLUDED.high,
//     low = EXCLUDED.low,
//     close = EXCLUDED.close,
//     vol = EXCLUDED.vol,
//     amount = EXCLUDED.amount,
//     change = EXCLUDED.change,
//     pct_chg = EXCLUDED.pct_chg,
//     pre_close = EXCLUDED.pre_close;
