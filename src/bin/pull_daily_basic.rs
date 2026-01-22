//! Pull all daily_basic data from Tushare and insert into database.
//! Usage: cargo run --bin pull_daily_basic

use reqwest::Client;
use serde_json::json;
use sqlx::{PgPool, Row, postgres::PgRow};
use std::env;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Hardcoded Tushare token
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    // Hardcoded DATABASE_URL as requested
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";

    // Connect to Postgres
    let pool = PgPool::connect(database_url).await?;

    // Tushare API endpoint
    let url = "https://api.tushare.pro";

    // Step 1: Get trading calendar from Tushare (SSE, since 20150101)
    let calendar_payload = json!({
        "api_name": "trade_cal",
        "token": tushare_token,
        "params": {
            "exchange": "SSE",
            "start_date": "20150101",
            "end_date": chrono::Local::now().format("%Y%m%d").to_string()
        },
        "fields": "cal_date,is_open"
    });

    let client = Client::new();
    let cal_resp = client
        .post(url)
        .json(&calendar_payload)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let cal_items = cal_resp["data"]["items"]
        .as_array()
        .expect("No calendar items in response");

    // Collect all open trading days
    let mut trading_days = Vec::new();
    for item in cal_items {
        let cal_date = item[0].as_str().unwrap();
        let is_open = match &item[1] {
            v if v.is_u64() => v.as_u64().unwrap() == 1,
            v if v.as_str().is_some() => v.as_str().unwrap() == "1",
            _ => false,
        };
        if is_open {
            trading_days.push(cal_date.to_string());
        }
    }
    println!("Found {} trading days since 20150101", trading_days.len());

    // Step 1b: Get max(trade_date) from daily_basic table
    let max_trade_date_row = sqlx::query("SELECT MAX(trade_date) FROM daily_basic")
        .fetch_one(&pool)
        .await?;
    let max_trade_date: Option<String> = max_trade_date_row.try_get(0)?;
    println!("Max trade_date in daily_basic: {:?}", max_trade_date);

    // Step 1c: Filter trading_days to only those after max_trade_date
    let filtered_trading_days: Vec<String> = match max_trade_date {
        Some(ref max_date) => trading_days
            .iter()
            .filter(|d| *d > max_date)
            .cloned()
            .collect(),
        None => trading_days.clone(),
    };
    println!("Will pull {} trading days", filtered_trading_days.len());

    // Step 2: Pull daily_basic for each trading day and insert
    let mut total_inserted = 0;
    for (idx, trade_date) in filtered_trading_days.iter().enumerate() {
        let payload = json!({
            "api_name": "daily_basic",
            "token": tushare_token,
            "params": { "trade_date": trade_date },
            "fields": "" // all fields
        });

        let resp = client
            .post(url)
            .json(&payload)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        // Fix: Avoid borrowing a temporary by using owned Vec for fallback
        let fields_vec = resp["data"]["fields"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let items_vec = resp["data"]["items"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let fields = &fields_vec;
        let items = &items_vec;

        if items.is_empty() {
            println!(
                "[{}/{}] {}: No daily_basic data",
                idx + 1,
                trading_days.len(),
                trade_date
            );
            continue;
        }

        let field_names: Vec<&str> = fields.iter().map(|v| v.as_str().unwrap()).collect();
        let placeholders: Vec<String> =
            (1..=field_names.len()).map(|i| format!("${}", i)).collect();
        let insert_sql = format!(
            "INSERT INTO daily_basic ({}) VALUES ({}) ON CONFLICT DO NOTHING",
            field_names.join(","),
            placeholders.join(",")
        );

        for item in items {
            let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();
            let mut query = sqlx::query(&insert_sql);
            for (idx, v) in params.iter().enumerate() {
                let field = field_names[idx];
                match field {
                    "close" | "turnover_rate" | "turnover_rate_f" | "volume_ratio" | "pe"
                    | "pe_ttm" | "pb" | "ps" | "ps_ttm" | "dv_ratio" | "dv_ttm" | "total_share"
                    | "float_share" | "free_share" | "total_mv" | "circ_mv" => {
                        if v.is_null() {
                            query = query.bind(None::<f64>);
                        } else if let Some(f) = v.as_f64() {
                            query = query.bind(f);
                        } else if let Some(s) = v.as_str() {
                            query = query.bind(s.parse::<f64>().ok());
                        } else {
                            query = query.bind(None::<f64>);
                        }
                    }
                    "ts_code" | "trade_date" => {
                        if v.is_null() {
                            query = query.bind(None::<String>);
                        } else if let Some(s) = v.as_str() {
                            query = query.bind(s);
                        } else {
                            query = query.bind(v.to_string());
                        }
                    }
                    _ => {
                        if v.is_null() {
                            query = query.bind(None::<String>);
                        } else if let Some(s) = v.as_str() {
                            query = query.bind(s);
                        } else if let Some(i) = v.as_i64() {
                            query = query.bind(i.to_string());
                        } else if let Some(f) = v.as_f64() {
                            query = query.bind(f.to_string());
                        } else {
                            query = query.bind(v.to_string());
                        }
                    }
                }
            }
            let result = query.execute(&pool).await?;
            total_inserted += result.rows_affected();
        }
        println!(
            "[{}/{}] {}: Inserted {} records",
            idx + 1,
            trading_days.len(),
            trade_date,
            items.len()
        );
    }

    println!("Inserted total {} rows into daily_basic", total_inserted);
    Ok(())
}
