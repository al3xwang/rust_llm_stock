use reqwest::Client;
use serde_json::json;
use sqlx::{PgPool, Row};
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Hardcoded Tushare token and database URL
    let tushare_token = "43748a5da1339b43a4956da729ae791f4e25ce9a15a356060658ffe6";
    let database_url = "postgresql://postgres:12341234@127.0.0.1:5432/research";

    // Connect to Postgres
    let pool = PgPool::connect(database_url).await?;

    // Ensure all required columns exist (add missing columns if needed)
    // Run each ALTER TABLE as a separate query to avoid "cannot insert multiple commands" error
    sqlx::query(r#"ALTER TABLE IF EXISTS tdx_member ADD COLUMN IF NOT EXISTS trade_date TEXT;"#)
        .execute(&pool)
        .await
        .ok();
    sqlx::query(r#"ALTER TABLE IF EXISTS tdx_member ADD COLUMN IF NOT EXISTS con_code TEXT;"#)
        .execute(&pool)
        .await
        .ok();
    sqlx::query(r#"ALTER TABLE IF EXISTS tdx_member ADD COLUMN IF NOT EXISTS con_name TEXT;"#)
        .execute(&pool)
        .await
        .ok();

    // Create table if not exists with correct unique constraint
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS tdx_member (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            con_code TEXT NOT NULL,
            con_name TEXT,
            CONSTRAINT tdx_member_unique UNIQUE (ts_code, trade_date, con_code)
        );
        "#,
    )
    .execute(&pool)
    .await?;

    // Prepare Tushare API request
    let url = "https://api.tushare.pro";
    let payload = json!({
        "api_name": "tdx_member",
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

    let fields = &fields_vec;
    let items = &items_vec;

    if items.is_empty() {
        println!("No tdx_member data returned from Tushare.");
        return Ok(());
    }

    let field_names: Vec<&str> = fields.iter().map(|v| v.as_str().unwrap()).collect();
    let placeholders: Vec<String> = (1..=field_names.len()).map(|i| format!("${}", i)).collect();
    let insert_sql = format!(
        "INSERT INTO tdx_member ({}) VALUES ({}) ON CONFLICT (ts_code, trade_date, con_code) DO UPDATE SET {}",
        field_names.join(","),
        placeholders.join(","),
        field_names
            .iter()
            .filter(|&&f| f != "ts_code" && f != "trade_date" && f != "con_code")
            .map(|f| format!("{} = EXCLUDED.{}", f, f))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut total_inserted = 0;
    for item in items {
        let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();
        let mut query = sqlx::query(&insert_sql);
        for v in params {
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
        let result = query.execute(&pool).await?;
        total_inserted += result.rows_affected();
    }

    println!("Inserted/updated {} rows into tdx_member", total_inserted);
    Ok(())
}
