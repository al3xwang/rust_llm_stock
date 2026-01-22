use anyhow::{Context, Result};
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_postgres::{NoTls, Client};
use std::env;

#[derive(Debug, Deserialize)]
struct TushareResponse<T> {
    code: i32,
    msg: Option<String>,
    data: Option<TushareData<T>>,
}

#[derive(Debug, Deserialize)]
struct TushareData<T> {
    fields: Vec<String>,
    items: Vec<T>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StockDaily {
    ts_code: String,
    trade_date: String,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    pre_close: Option<f64>,
    change: Option<f64>,
    pct_chg: Option<f64>,
    vol: Option<f64>,
    amount: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StockDailyAdjusted {
    ts_code: String,
    trade_date: String,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    pre_close: Option<f64>,
    change: Option<f64>,
    pct_chg: Option<f64>,
    vol: Option<f64>,
    amount: Option<f64>,
    adj_factor: Option<f64>,
}

struct TushareClient {
    api_token: String,
    base_url: String,
    client: reqwest::Client,
}

impl TushareClient {
    fn new(api_token: String) -> Self {
        Self {
            api_token,
            base_url: "http://api.tushare.pro".to_string(),
            client: reqwest::Client::new(),
        }
    }

    async fn get_stock_daily(&self, ts_code: &str, start_date: &str, end_date: &str) -> Result<Vec<StockDaily>> {
        let params = json!({
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date
        });
        let request_body = json!({
            "api_name": "daily",
            "token": self.api_token,
            "params": params,
            "fields": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        });
        let response = self
            .client
            .post(&self.base_url)
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to Tushare API")?;
        if !response.status().is_success() {
            anyhow::bail!("API request failed with status: {}", response.status());
        }
        let api_response: TushareResponse<Vec<serde_json::Value>> = response
            .json()
            .await
            .context("Failed to parse API response")?;
        if api_response.code != 0 {
            anyhow::bail!(
                "API returned error code {}: {}",
                api_response.code,
                api_response.msg.unwrap_or_default()
            );
        }
        let data = api_response.data.context("No data in response")?;
        let mut results = Vec::new();
        for item_vec in data.items {
            let daily = StockDaily {
                ts_code: item_vec.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                trade_date: item_vec.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                open: item_vec.get(2).and_then(|v| v.as_f64()),
                high: item_vec.get(3).and_then(|v| v.as_f64()),
                low: item_vec.get(4).and_then(|v| v.as_f64()),
                close: item_vec.get(5).and_then(|v| v.as_f64()),
                pre_close: item_vec.get(6).and_then(|v| v.as_f64()),
                change: item_vec.get(7).and_then(|v| v.as_f64()),
                pct_chg: item_vec.get(8).and_then(|v| v.as_f64()),
                vol: item_vec.get(9).and_then(|v| v.as_f64()),
                amount: item_vec.get(10).and_then(|v| v.as_f64()),
            };
            results.push(daily);
        }
        Ok(results)
    }

    async fn get_stock_daily_adjusted(&self, ts_code: &str, start_date: &str, end_date: &str) -> Result<Vec<StockDailyAdjusted>> {
        let params = json!({
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "adj": "qfq"  // Forward-adjusted prices (前复权)
        });
        let request_body = json!({
            "api_name": "daily",
            "token": self.api_token,
            "params": params,
            "fields": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor"
        });
        let response = self
            .client
            .post(&self.base_url)
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to Tushare API")?;
        if !response.status().is_success() {
            anyhow::bail!("API request failed with status: {}", response.status());
        }
        let api_response: TushareResponse<Vec<serde_json::Value>> = response
            .json()
            .await
            .context("Failed to parse API response")?;
        if api_response.code != 0 {
            anyhow::bail!(
                "API returned error code {}: {}",
                api_response.code,
                api_response.msg.unwrap_or_default()
            );
        }
        let data = api_response.data.context("No data in response")?;
        let mut results = Vec::new();
        for item_vec in data.items {
            let daily = StockDailyAdjusted {
                ts_code: item_vec.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                trade_date: item_vec.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                open: item_vec.get(2).and_then(|v| v.as_f64()),
                high: item_vec.get(3).and_then(|v| v.as_f64()),
                low: item_vec.get(4).and_then(|v| v.as_f64()),
                close: item_vec.get(5).and_then(|v| v.as_f64()),
                pre_close: item_vec.get(6).and_then(|v| v.as_f64()),
                change: item_vec.get(7).and_then(|v| v.as_f64()),
                pct_chg: item_vec.get(8).and_then(|v| v.as_f64()),
                vol: item_vec.get(9).and_then(|v| v.as_f64()),
                amount: item_vec.get(10).and_then(|v| v.as_f64()),
                adj_factor: item_vec.get(11).and_then(|v| v.as_f64()),
            };
            results.push(daily);
        }
        Ok(results)
    }
}

async fn create_stock_daily_table(client: &Client) -> Result<()> {
    let create_table_sql = r#"
        CREATE TABLE IF NOT EXISTS stock_daily (
            ts_code VARCHAR(20) NOT NULL,
            trade_date VARCHAR(8) NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            pre_close DOUBLE PRECISION,
            change DOUBLE PRECISION,
            pct_chg DOUBLE PRECISION,
            vol DOUBLE PRECISION,
            amount DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code ON stock_daily(ts_code);
        CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date ON stock_daily(trade_date);
    "#;
    client.batch_execute(create_table_sql).await.context("Failed to create stock_daily table")?;
    println!("✓ Table 'stock_daily' created successfully");
    Ok(())
}

async fn insert_stock_daily(client: &Client, records: Vec<StockDaily>) -> Result<usize> {
    let insert_sql = r#"
        INSERT INTO stock_daily (
            ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        ON CONFLICT (ts_code, trade_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            pre_close = EXCLUDED.pre_close,
            change = EXCLUDED.change,
            pct_chg = EXCLUDED.pct_chg,
            vol = EXCLUDED.vol,
            amount = EXCLUDED.amount,
            updated_at = CURRENT_TIMESTAMP
    "#;
    let mut inserted = 0;
    for rec in records {
        let rows = client
            .execute(
                insert_sql,
                &[
                    &rec.ts_code,
                    &rec.trade_date,
                    &rec.open,
                    &rec.high,
                    &rec.low,
                    &rec.close,
                    &rec.pre_close,
                    &rec.change,
                    &rec.pct_chg,
                    &rec.vol,
                    &rec.amount,
                ],
            )
            .await
            .context("Failed to insert stock daily data")?;
        if rows > 0 {
            inserted += 1;
        }
    }
    Ok(inserted)
}

async fn create_stock_daily_adjusted_table(client: &Client) -> Result<()> {
    let create_table_sql = r#"
        CREATE TABLE IF NOT EXISTS stock_daily_adjusted (
            ts_code VARCHAR(20) NOT NULL,
            trade_date VARCHAR(8) NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            pre_close DOUBLE PRECISION,
            volume DOUBLE PRECISION NOT NULL,
            amount DOUBLE PRECISION,
            change DOUBLE PRECISION,
            pct_chg DOUBLE PRECISION,
            adj_factor DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_ts_code ON stock_daily_adjusted(ts_code);
        CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_trade_date ON stock_daily_adjusted(trade_date);
        CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_ts_code_date ON stock_daily_adjusted(ts_code, trade_date DESC);
    "#;
    client.batch_execute(create_table_sql).await.context("Failed to create stock_daily_adjusted table")?;
    println!("✓ Table 'stock_daily_adjusted' created successfully");
    Ok(())
}

async fn insert_stock_daily_adjusted(client: &Client, records: Vec<StockDailyAdjusted>) -> Result<usize> {
    let insert_sql = r#"
        INSERT INTO stock_daily_adjusted (
            ts_code, trade_date, open, high, low, close, pre_close, volume, amount, change, pct_chg, adj_factor
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        )
        ON CONFLICT (ts_code, trade_date)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            pre_close = EXCLUDED.pre_close,
            volume = EXCLUDED.volume,
            amount = EXCLUDED.amount,
            change = EXCLUDED.change,
            pct_chg = EXCLUDED.pct_chg,
            adj_factor = EXCLUDED.adj_factor,
            updated_at = CURRENT_TIMESTAMP
    "#;
    let mut inserted = 0;
    for rec in records {
        let rows = client
            .execute(
                insert_sql,
                &[
                    &rec.ts_code,
                    &rec.trade_date,
                    &rec.open,
                    &rec.high,
                    &rec.low,
                    &rec.close,
                    &rec.pre_close,
                    &rec.vol,
                    &rec.amount,
                    &rec.change,
                    &rec.pct_chg,
                    &rec.adj_factor,
                ],
            )
            .await
            .context("Failed to insert stock daily adjusted data")?;
        if rows > 0 {
            inserted += 1;
        }
    }
    Ok(inserted)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    println!("╔════════════════════════════════════════════════╗");
    println!("║     Tushare Stock Daily Data Ingestion        ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    let api_token = env::var("TUSHARE_TOKEN").context("TUSHARE_TOKEN environment variable not set. Please set it with your Tushare API token")?;
    let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| "postgresql://alex:123456@localhost/research".to_string());
    println!("Connecting to database...");
    let (client, connection) = tokio_postgres::connect(&db_url, NoTls).await.context("Failed to connect to database")?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Database connection error: {}", e);
        }
    });
    println!("✓ Connected to database");
    println!();
    println!("Creating tables if not exists...");
    create_stock_daily_table(&client).await?;
    create_stock_daily_adjusted_table(&client).await?;
    println!();
    let tushare = TushareClient::new(api_token);
    // TODO: Replace with your stock list and date range
    let stocks = vec!["000001.SZ", "600000.SH"];
    let start_date = "19901219";
    let end_date = "20251223";
    let mut total_inserted = 0;
    let mut total_adjusted_inserted = 0;
    for ts_code in stocks {
        println!("Fetching {} daily data (unadjusted)...", ts_code);
        match tushare.get_stock_daily(ts_code, start_date, end_date).await {
            Ok(records) => {
                println!("  ✓ Fetched {} records", records.len());
                if !records.is_empty() {
                    println!("  Inserting into database...");
                    match insert_stock_daily(&client, records).await {
                        Ok(count) => {
                            println!("  ✓ Inserted/Updated {} records", count);
                            total_inserted += count;
                        }
                        Err(e) => {
                            eprintln!("  ✗ Failed to insert data: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("  ✗ Failed to fetch data: {}", e);
            }
        }
        
        // Fetch forward-adjusted prices
        println!("Fetching {} daily data (forward-adjusted)...", ts_code);
        match tushare.get_stock_daily_adjusted(ts_code, start_date, end_date).await {
            Ok(records) => {
                println!("  ✓ Fetched {} adjusted records", records.len());
                if !records.is_empty() {
                    println!("  Inserting into stock_daily_adjusted table...");
                    match insert_stock_daily_adjusted(&client, records).await {
                        Ok(count) => {
                            println!("  ✓ Inserted/Updated {} adjusted records", count);
                            total_adjusted_inserted += count;
                        }
                        Err(e) => {
                            eprintln!("  ✗ Failed to insert adjusted data: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("  ✗ Failed to fetch adjusted data: {}", e);
            }
        }
        
        println!();
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    println!();
    println!("╔════════════════════════════════════════════════╗");
    println!("║            Ingestion Complete                  ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("Total unadjusted records inserted/updated: {}", total_inserted);
    println!("Total adjusted records inserted/updated: {}", total_adjusted_inserted);
    Ok(())
}
