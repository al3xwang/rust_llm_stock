use reqwest::Client;
use serde_json::json;
use sqlx::{PgPool, Row};
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Step 1: Get trading calendar from Tushare (SSE, since 20150101)
    let client = Client::new();
    let url = "https://api.tushare.pro";
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";
    let pool = PgPool::connect(database_url).await?;

    // First, make a sample API call to get field names for dynamic table creation
    let sample_payload = json!({
        "api_name": "dc_daily",
        "token": tushare_token,
        "params": { "trade_date": "20240101" }, // recent date to get schema
        "fields": "" // all fields
    });

    let sample_resp = client
        .post(url)
        .json(&sample_payload)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let api_fields = sample_resp["data"]["fields"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let sample_items = sample_resp["data"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    // Infer data types from sample rows (check all rows, not just first)
    let mut field_types: Vec<String> = vec!["TEXT".to_string(); api_fields.len()];

    for item in &sample_items {
        if let Some(values) = item.as_array() {
            for (i, value) in values.iter().enumerate() {
                if i < field_types.len() && field_types[i] == "TEXT" {
                    // Once we find a number, mark this field as DOUBLE PRECISION
                    if value.is_number() {
                        field_types[i] = "DOUBLE PRECISION".to_string();
                    }
                }
            }
        }
    }

    // Default to DOUBLE PRECISION for fields with numeric-sounding names
    for (i, field) in api_fields.iter().enumerate() {
        if i < field_types.len() && field_types[i] == "TEXT" {
            let field_name = field.as_str().unwrap().to_lowercase();
            let is_likely_numeric = field_name.contains("count")
                || field_name.contains("share")
                || field_name.contains("mv")
                || field_name.contains("volume")
                || field_name.contains("vol")
                || field_name.contains("amount")
                || field_name.contains("num")
                || field_name.contains("ratio")
                || field_name.contains("rate")
                || field_name.contains("pct")
                || field_name.contains("change")
                || field_name.contains("value")
                || field_name.contains("price")
                || field_name.contains("idx_")
                || field_name.contains("swing")
                || field_name == "open"
                || field_name == "high"
                || field_name == "low"
                || field_name == "close"
                || field_name == "open_"
                || field_name == "high_"
                || field_name == "low_"
                || field_name == "close_";

            if is_likely_numeric {
                field_types[i] = "DOUBLE PRECISION".to_string();
            }
        }
    }

    // Build dynamic CREATE TABLE with inferred types
    let mut column_defs = Vec::new();
    for (i, field) in api_fields.iter().enumerate() {
        let field_name = field.as_str().unwrap();
        let quoted_name = format!("\"{}\"", field_name);
        let data_type = field_types.get(i).map(|s| s.as_str()).unwrap_or("TEXT");

        if field_name == "ts_code" || field_name == "trade_date" {
            column_defs.push(format!("{} TEXT NOT NULL", quoted_name));
        } else {
            column_defs.push(format!("{} {}", quoted_name, data_type));
        }
    }
    column_defs.push("PRIMARY KEY (\"ts_code\", \"trade_date\")".to_string());

    let create_table_sql = format!(
        "CREATE TABLE IF NOT EXISTS dc_daily ({})",
        column_defs.join(", ")
    );

    sqlx::query(&create_table_sql).execute(&pool).await?;
    println!(
        "dc_daily table created/verified with {} columns",
        api_fields.len()
    );

    // Fetch trading calendar
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

    let cal_resp = client
        .post(url)
        .json(&calendar_payload)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let cal_items = cal_resp["data"]["items"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let mut trading_days = Vec::new();
    for item in &cal_items {
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

    // Get max(trade_date) from dc_daily table for smart backward backfill
    let max_trade_date_row = sqlx::query("SELECT MAX(trade_date) FROM dc_daily")
        .fetch_optional(&pool)
        .await?;
    let max_trade_date: Option<String> = max_trade_date_row.and_then(|row| row.get(0));
    println!("Max trade_date in dc_daily: {:?}", max_trade_date);

    // Reverse trading days and process backward from today until:
    // 1. 15 consecutive days with no data returned, OR
    // 2. Reached max(trade_date) from database
    let mut trading_days_reversed: Vec<String> = trading_days.clone();
    trading_days_reversed.reverse();

    let trading_days_to_process: Vec<String> = if let Some(max_date) = max_trade_date.clone() {
        // Skip trading days up to and including max_date
        trading_days_reversed
            .into_iter()
            .take_while(|d| d > &max_date)
            .collect()
    } else {
        trading_days_reversed
    };

    println!(
        "Processing up to {} trading days backward for dc_daily (until 15 empty days or max_date reached)",
        trading_days_to_process.len()
    );

    // Step 2: Pull dc_daily for each trading day backward until 15 consecutive empty days
    let mut total_inserted = 0;
    let mut consecutive_empty_days = 0;
    const MAX_CONSECUTIVE_EMPTY: usize = 15;

    for (idx, trade_date) in trading_days_to_process.iter().enumerate() {
        let payload = json!({
            "api_name": "dc_daily",
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
            consecutive_empty_days += 1;
            println!(
                "[{}/{}] {}: No dc_daily data ({}/{} empty days)",
                idx + 1,
                trading_days_to_process.len(),
                trade_date,
                consecutive_empty_days,
                MAX_CONSECUTIVE_EMPTY
            );

            // Stop if we've seen 15 consecutive empty days
            if consecutive_empty_days >= MAX_CONSECUTIVE_EMPTY {
                println!(
                    "\n⏹️  Stopping: Reached {} consecutive days with no data. All historical data processed.",
                    MAX_CONSECUTIVE_EMPTY
                );
                break;
            }
            continue;
        } else {
            // Reset counter when we find data
            consecutive_empty_days = 0;
        }

        // Dynamically construct SQL to handle all field names (quote them all)
        let field_names: Vec<String> = fields
            .iter()
            .map(|v| {
                let fname = v.as_str().unwrap();
                format!("\"{}\"", fname)
            })
            .collect();

        let placeholders: Vec<String> =
            (1..=field_names.len()).map(|i| format!("${}", i)).collect();

        let update_fields: Vec<String> = field_names
            .iter()
            .filter(|f| f.as_str() != "\"ts_code\"" && f.as_str() != "\"trade_date\"")
            .map(|f| format!("{} = EXCLUDED.{}", f, f))
            .collect();

        let insert_sql = format!(
            "INSERT INTO dc_daily ({}) VALUES ({}) ON CONFLICT (\"ts_code\", \"trade_date\") DO UPDATE SET {}",
            field_names.join(","),
            placeholders.join(","),
            update_fields.join(", ")
        );

        let mut count = 0;
        for item in items {
            let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();

            let mut query = sqlx::query(&insert_sql);
            for param in &params {
                match param {
                    serde_json::Value::String(s) => query = query.bind(s),
                    serde_json::Value::Number(n) if n.is_f64() => query = query.bind(n.as_f64()),
                    serde_json::Value::Number(n) if n.is_i64() => query = query.bind(n.as_i64()),
                    serde_json::Value::Null => query = query.bind(Option::<String>::None),
                    _ => query = query.bind(param.to_string()),
                }
            }

            query.execute(&pool).await?;
            count += 1;
        }

        total_inserted += count;
        println!(
            "[{}/{}] {}: Inserted {} rows (empty streak: 0/{})\n",
            idx + 1,
            trading_days_to_process.len(),
            trade_date,
            count,
            MAX_CONSECUTIVE_EMPTY
        );

        // Rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
    }

    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("✅ dc_daily ingestion complete!");
    println!("   Inserted: {} rows", total_inserted);
    println!("   Consecutive empty days limit: {}", MAX_CONSECUTIVE_EMPTY);
    println!("   Processing stopped when limit reached or max_date reached");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════\n"
    );
    Ok(())
}
