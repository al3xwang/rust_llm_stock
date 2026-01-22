use anyhow::{Context, Result};
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_postgres::{NoTls, Client};
use std::env;

/// Tushare API response structure
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

/// Stock basic information from tushare
#[derive(Debug, Serialize, Deserialize)]
struct StockBasic {
    ts_code: String,           // TS代码
    symbol: String,            // 股票代码
    name: String,              // 股票名称
    area: Option<String>,      // 地域
    industry: Option<String>,  // 所属行业
    fullname: Option<String>,  // 股票全称
    enname: Option<String>,    // 英文全称
    cnspell: Option<String>,   // 拼音缩写
    market: String,            // 市场类型（主板/创业板/科创板等）
    exchange: Option<String>,  // 交易所代码
    curr_type: Option<String>, // 交易货币
    list_status: String,       // 上市状态 L上市 D退市 P暂停上市
    list_date: Option<String>, // 上市日期
    delist_date: Option<String>, // 退市日期
    is_hs: Option<String>,     // 是否沪深港通标的，N否 H沪股通 S深股通
}

/// Tushare API client
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

    /// Fetch stock basic data
    async fn get_stock_basic(&self, list_status: Option<&str>) -> Result<Vec<StockBasic>> {
        let params = if let Some(status) = list_status {
            json!({ "list_status": status })
        } else {
            json!({})
        };

        let request_body = json!({
            "api_name": "stock_basic",
            "token": self.api_token,
            "params": params,
            "fields": "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs"
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

        // Convert items to StockBasic
        let mut results = Vec::new();
        for item_vec in data.items {
            let stock_basic = StockBasic {
                ts_code: item_vec.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                symbol: item_vec.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                name: item_vec.get(2).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                area: item_vec.get(3).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                industry: item_vec.get(4).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                fullname: item_vec.get(5).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                enname: item_vec.get(6).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                cnspell: item_vec.get(7).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                market: item_vec.get(8).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                exchange: item_vec.get(9).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                curr_type: item_vec.get(10).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                list_status: item_vec.get(11).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                list_date: item_vec.get(12).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                delist_date: item_vec.get(13).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                is_hs: item_vec.get(14).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
            };
            
            results.push(stock_basic);
        }

        Ok(results)
    }
}

/// Database operations
async fn create_stock_basic_table(client: &Client) -> Result<()> {
    let create_table_sql = r#"
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code VARCHAR(20) PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            name VARCHAR(100) NOT NULL,
            area VARCHAR(50),
            industry VARCHAR(100),
            fullname VARCHAR(200),
            enname VARCHAR(200),
            cnspell VARCHAR(50),
            market VARCHAR(20) NOT NULL,
            exchange VARCHAR(20),
            curr_type VARCHAR(10),
            list_status VARCHAR(1) NOT NULL,
            list_date VARCHAR(20),
            delist_date VARCHAR(20),
            is_hs VARCHAR(1),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_stock_basic_symbol ON stock_basic(symbol);
        CREATE INDEX IF NOT EXISTS idx_stock_basic_market ON stock_basic(market);
        CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
        CREATE INDEX IF NOT EXISTS idx_stock_basic_list_status ON stock_basic(list_status);
        CREATE INDEX IF NOT EXISTS idx_stock_basic_list_date ON stock_basic(list_date);
    "#;

    client
        .batch_execute(create_table_sql)
        .await
        .context("Failed to create stock_basic table")?;

    println!("✓ Table 'stock_basic' created successfully");
    Ok(())
}

async fn insert_stock_data(client: &Client, stocks: Vec<StockBasic>) -> Result<usize> {
    let insert_sql = r#"
        INSERT INTO stock_basic (
            ts_code, symbol, name, area, industry, fullname, enname, 
            cnspell, market, exchange, curr_type, list_status, 
            list_date, delist_date, is_hs
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        )
        ON CONFLICT (ts_code) 
        DO UPDATE SET
            symbol = EXCLUDED.symbol,
            name = EXCLUDED.name,
            area = EXCLUDED.area,
            industry = EXCLUDED.industry,
            fullname = EXCLUDED.fullname,
            enname = EXCLUDED.enname,
            cnspell = EXCLUDED.cnspell,
            market = EXCLUDED.market,
            exchange = EXCLUDED.exchange,
            curr_type = EXCLUDED.curr_type,
            list_status = EXCLUDED.list_status,
            list_date = EXCLUDED.list_date,
            delist_date = EXCLUDED.delist_date,
            is_hs = EXCLUDED.is_hs,
            updated_at = CURRENT_TIMESTAMP
    "#;

    let mut inserted = 0;

    for stock in stocks {
        let rows = client
            .execute(
                insert_sql,
                &[
                    &stock.ts_code,
                    &stock.symbol,
                    &stock.name,
                    &stock.area,
                    &stock.industry,
                    &stock.fullname,
                    &stock.enname,
                    &stock.cnspell,
                    &stock.market,
                    &stock.exchange,
                    &stock.curr_type,
                    &stock.list_status,
                    &stock.list_date,
                    &stock.delist_date,
                    &stock.is_hs,
                ],
            )
            .await
            .context("Failed to insert stock data")?;

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
    println!("║     Tushare Stock Basic Data Ingestion        ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();

    // Get Tushare API token from environment
    let api_token = env::var("TUSHARE_TOKEN")
        .context("TUSHARE_TOKEN environment variable not set. Please set it with your Tushare API token")?;

    // Get database connection string - default to localhost
    let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
        "postgresql://alex:123456@localhost/research".to_string()
    });

    println!("Connecting to database: {}", db_url.split('@').last().unwrap_or(""));
    let (client, connection) = tokio_postgres::connect(&db_url, NoTls)
        .await
        .context("Failed to connect to database")?;

    // Spawn connection
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Database connection error: {}", e);
        }
    });

    println!("✓ Connected to database");
    println!();

    // Create table if not exists
    println!("Creating table if not exists...");
    create_stock_basic_table(&client).await?;
    println!();

    // Initialize Tushare client
    let tushare = TushareClient::new(api_token);

    // Fetch listed stocks
    println!("Fetching listed stocks (L)...");
    match tushare.get_stock_basic(Some("L")).await {
        Ok(stocks) => {
            println!("  ✓ Fetched {} stocks", stocks.len());
            
            if !stocks.is_empty() {
                println!("  Inserting into database...");
                match insert_stock_data(&client, stocks).await {
                    Ok(count) => {
                        println!("  ✓ Inserted/Updated {} records", count);
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

    println!();
    println!("╔════════════════════════════════════════════════╗");
    println!("║            Ingestion Complete                  ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();

    // Query summary
    let count_query = "SELECT market, list_status, COUNT(*) as count FROM stock_basic GROUP BY market, list_status ORDER BY count DESC";
    let rows = client.query(count_query, &[]).await?;
    
    println!("Stock Summary:");
    println!("─────────────────────────────────────────────────");
    for row in rows {
        let market: String = row.get(0);
        let status: String = row.get(1);
        let count: i64 = row.get(2);
        let status_name = match status.as_str() {
            "L" => "Listed",
            "D" => "Delisted",
            "P" => "Paused",
            _ => "Unknown"
        };
        println!("  {:<15} {:<10} : {} stocks", market, status_name, count);
    }

    // Industry summary
    println!();
    let industry_query = "SELECT industry, COUNT(*) as count FROM stock_basic WHERE list_status = 'L' AND industry IS NOT NULL GROUP BY industry ORDER BY count DESC LIMIT 10";
    let rows = client.query(industry_query, &[]).await?;
    
    println!("Top 10 Industries:");
    println!("─────────────────────────────────────────────────");
    for row in rows {
        let industry: String = row.get(0);
        let count: i64 = row.get(1);
        println!("  {:<30} : {} stocks", industry, count);
    }

    Ok(())
}
