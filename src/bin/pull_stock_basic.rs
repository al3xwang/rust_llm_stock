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

    // Ensure all columns exist in stock_basic table (add missing columns if needed)
    // Run each ALTER TABLE as a separate query to avoid "cannot insert multiple commands" error
    sqlx::query(r#"ALTER TABLE stock_basic ADD COLUMN IF NOT EXISTS act_name TEXT;"#)
        .execute(&pool)
        .await
        .ok();
    sqlx::query(r#"ALTER TABLE stock_basic ADD COLUMN IF NOT EXISTS act_ent_type TEXT;"#)
        .execute(&pool)
        .await
        .ok();

    // Create stock_basic table if not exists (fields per Tushare docs)
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            area TEXT,
            industry TEXT,
            fullname TEXT,
            enname TEXT,
            cnspell TEXT,
            market TEXT,
            exchange TEXT,
            curr_type TEXT,
            list_status TEXT,
            list_date TEXT,
            delist_date TEXT,
            is_hs TEXT,
            act_name TEXT,
            act_ent_type TEXT
        )
        "#,
    )
    .execute(&pool)
    .await?;

    // Prepare Tushare API request
    let url = "https://api.tushare.pro";
    let payload = json!({
        "api_name": "stock_basic",
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
        println!("No stock_basic data returned from Tushare.");
        return Ok(());
    }

    let field_names: Vec<&str> = fields.iter().map(|v| v.as_str().unwrap()).collect();
    let placeholders: Vec<String> = (1..=field_names.len()).map(|i| format!("${}", i)).collect();
    let insert_sql = format!(
        "INSERT INTO stock_basic ({}) VALUES ({}) ON CONFLICT (ts_code) DO UPDATE SET {}",
        field_names.join(","),
        placeholders.join(","),
        field_names
            .iter()
            .filter(|&&f| f != "ts_code")
            .map(|f| format!("{} = EXCLUDED.{}", f, f))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut total_inserted = 0;
    for item in items {
        let params: Vec<&serde_json::Value> = item.as_array().unwrap().iter().collect();
        let mut query = sqlx::query(&insert_sql);
        for (idx, v) in params.iter().enumerate() {
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

    println!("Inserted/updated {} rows into stock_basic", total_inserted);
    Ok(())
}
