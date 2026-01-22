use reqwest::Client;
use serde_json::json;
use sqlx::PgPool;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Hardcoded Tushare token and database URL
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";

    // Connect to Postgres
    let pool = PgPool::connect(database_url).await?;

    // Prepare Tushare API request to get schema
    let url = "https://api.tushare.pro";
    let payload = json!({
        "api_name": "dc_member",
        "token": tushare_token,
        "params": {},
        "fields": "" // all fields
    });

    let client = Client::new();
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
        println!("No dc_member data returned from Tushare.");
        return Ok(());
    }

    // Infer data types from sample rows (check all rows, not just first)
    let mut field_types: Vec<String> = vec!["TEXT".to_string(); fields_vec.len()];

    for item in &items_vec {
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
    for (i, field) in fields_vec.iter().enumerate() {
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
                || field_name == "close"
                || field_name == "high"
                || field_name == "low";

            if is_likely_numeric {
                field_types[i] = "DOUBLE PRECISION".to_string();
            }
        }
    }

    let mut column_defs = Vec::new();
    for (i, field) in fields_vec.iter().enumerate() {
        let field_name = field.as_str().unwrap();
        let quoted_name = format!("\"{}\"", field_name);
        let data_type = field_types.get(i).map(|s| s.as_str()).unwrap_or("TEXT");

        if field_name == "ts_code" || field_name == "trade_date" || field_name == "con_code" {
            column_defs.push(format!("{} TEXT NOT NULL", quoted_name));
        } else {
            column_defs.push(format!("{} {}", quoted_name, data_type));
        }
    }

    // Add unique constraint
    column_defs.push(
        "CONSTRAINT dc_member_unique UNIQUE (\"ts_code\", \"trade_date\", \"con_code\")"
            .to_string(),
    );

    let create_table_sql = format!(
        "CREATE TABLE IF NOT EXISTS dc_member ({})",
        column_defs.join(", ")
    );

    sqlx::query(&create_table_sql).execute(&pool).await?;
    println!("dc_member table created with {} columns", fields_vec.len());

    let fields = &fields_vec;
    let items = &items_vec;

    // Quote field names to handle SQL reserved keywords
    let field_names: Vec<String> = fields
        .iter()
        .map(|v| {
            let fname = v.as_str().unwrap();
            format!("\"{}\"", fname)
        })
        .collect();
    let placeholders: Vec<String> = (1..=field_names.len()).map(|i| format!("${}", i)).collect();
    let insert_sql = format!(
        "INSERT INTO dc_member ({}) VALUES ({}) ON CONFLICT (\"ts_code\", \"trade_date\", \"con_code\") DO UPDATE SET {}",
        field_names.join(","),
        placeholders.join(","),
        field_names
            .iter()
            .filter(|f| f.as_str() != "\"ts_code\""
                && f.as_str() != "\"trade_date\""
                && f.as_str() != "\"con_code\"")
            .map(|f| format!("{} = EXCLUDED.{}", f, f))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut total_inserted = 0;
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
        total_inserted += 1;
    }

    println!("Inserted/updated {} rows into dc_member", total_inserted);
    Ok(())
}
