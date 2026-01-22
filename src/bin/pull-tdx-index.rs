use chrono::{Duration as ChronoDuration, Local, NaiveDate};
use reqwest::Client;
use serde_json::json;
use sqlx::PgPool;
use std::thread;
use std::time::Duration;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::new();
    let url = "https://api.tushare.pro";
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";
    let pool = PgPool::connect(database_url).await?;

    // Create tdx_index table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS tdx_index (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            name TEXT,
            idx_type TEXT,
            idx_count DOUBLE PRECISION,
            total_share DOUBLE PRECISION,
            float_share DOUBLE PRECISION,
            total_mv DOUBLE PRECISION,
            float_mv DOUBLE PRECISION,
            PRIMARY KEY (ts_code, trade_date)
        )
        "#,
    )
    .execute(&pool)
    .await?;

    // Get most recent date in database (stop condition)
    let stop_date: Option<NaiveDate> = sqlx::query_scalar("SELECT MAX(trade_date) FROM tdx_index")
        .fetch_optional(&pool)
        .await?
        .flatten()
        .and_then(|s: String| NaiveDate::parse_from_str(&s, "%Y%m%d").ok());

    if let Some(ref max_date) = stop_date {
        println!(
            "✓ Existing data found up to {}",
            max_date.format("%Y-%m-%d")
        );
    } else {
        println!("⚠ No existing data - full historical pull");
    }

    // Start from today and work backward
    let curr_date = Local::now().naive_local().date();
    let mut current_date = curr_date;
    let mut total_inserted = 0;
    let mut consecutive_empty_days = 0;
    const MAX_EMPTY_DAYS: i32 = 15;

    println!(
        "Starting backward iteration from {}",
        curr_date.format("%Y-%m-%d")
    );
    print!("Progress: ");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    loop {
        // Stop condition 1: Reached existing data
        if let Some(ref max_date) = stop_date {
            if current_date <= *max_date {
                println!(
                    "\n✓ Reached existing data at {}",
                    max_date.format("%Y-%m-%d")
                );
                break;
            }
        }

        // Stop condition 2: 15 consecutive empty days
        if consecutive_empty_days >= MAX_EMPTY_DAYS {
            println!(
                "\n✓ Early stopping: {} consecutive days with no data",
                consecutive_empty_days
            );
            break;
        }

        let trade_date = current_date.format("%Y%m%d").to_string();

        // Pull data for this date
        let payload = json!({
            "api_name": "tdx_index",
            "token": tushare_token,
            "params": { "trade_date": &trade_date },
            "fields": ""
        });

        match pull_and_insert(&client, url, &pool, payload).await {
            Ok(inserted) => {
                if inserted > 0 {
                    consecutive_empty_days = 0; // Reset on success
                    total_inserted += inserted;
                    print!(".");
                } else {
                    consecutive_empty_days += 1;
                    print!("·");
                }
            }
            Err(e) => {
                consecutive_empty_days += 1;
                eprintln!("\n  ✗ error for {}: {}", trade_date, e);
                print!("·");
            }
        }

        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Move backward one day
        current_date = current_date
            .checked_sub_signed(ChronoDuration::days(1))
            .unwrap();

        // Rate limiting - 300ms between requests
        thread::sleep(Duration::from_millis(300));
    }

    println!("\n✅ Inserted {} total rows into tdx_index", total_inserted);
    Ok(())
}

async fn pull_and_insert(
    client: &Client,
    url: &str,
    pool: &PgPool,
    payload: serde_json::Value,
) -> anyhow::Result<u64> {
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let fields_vec = resp["data"]["fields"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let items_vec = resp["data"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    if items_vec.is_empty() {
        return Ok(0);
    }

    let field_names: Vec<&str> = fields_vec.iter().map(|v| v.as_str().unwrap()).collect();
    let placeholders: Vec<String> = (1..=field_names.len()).map(|i| format!("${}", i)).collect();
    let insert_sql = format!(
        "INSERT INTO tdx_index ({}) VALUES ({}) ON CONFLICT (ts_code, trade_date) DO UPDATE SET {}",
        field_names.join(","),
        placeholders.join(","),
        field_names
            .iter()
            .filter(|&&f| f != "ts_code" && f != "trade_date")
            .map(|f| format!("{} = EXCLUDED.{}", f, f))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut total_affected = 0u64;
    for item in &items_vec {
        let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();
        let mut query = sqlx::query(&insert_sql);
        for (idx, v) in params.iter().enumerate() {
            let field = field_names[idx];
            match field {
                "idx_count" | "total_share" | "float_share" | "total_mv" | "float_mv" => {
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
                _ => {
                    if v.is_null() {
                        query = query.bind(None::<String>);
                    } else if let Some(s) = v.as_str() {
                        query = query.bind(s);
                    } else {
                        query = query.bind(v.to_string());
                    }
                }
            }
        }
        let result = query.execute(pool).await?;
        total_affected += result.rows_affected();
    }

    Ok(total_affected)
}
