use anyhow::{Context, Result};
use chrono::NaiveDate;
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

/// Index basic information from tushare
#[derive(Debug, Serialize, Deserialize)]
struct IndexBasic {
    ts_code: String,        // TS代码
    name: String,           // 简称
    fullname: Option<String>, // 全称
    market: String,         // 市场
    publisher: Option<String>, // 发布方
    index_type: Option<String>, // 指数类型
    category: Option<String>, // 指数类别
    base_date: Option<String>, // 基期
    base_point: Option<f64>, // 基点
    list_date: Option<String>, // 发布日期
    weight_rule: Option<String>, // 加权方式
    desc: Option<String>,   // 描述
    exp_date: Option<String>, // 终止日期
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

    /// Fetch index basic data
    async fn get_index_basic(&self, market: Option<&str>) -> Result<Vec<IndexBasic>> {
        let params = if let Some(m) = market {
            json!({ "market": m })
        } else {
            json!({})
        };

        let request_body = json!({
            "api_name": "index_basic",
            "token": self.api_token,
            "params": params,
            "fields": "ts_code,name,fullname,market,publisher,index_type,category,base_date,base_point,list_date,weight_rule,desc,exp_date"
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

        let data = api_response
            .data
            .context("No data in response")?;

        // Convert items to IndexBasic
        let mut results = Vec::new();
        for item_vec in data.items {
            let index_basic = IndexBasic {
                ts_code: item_vec.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                name: item_vec.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                fullname: item_vec.get(2).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                market: item_vec.get(3).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                publisher: item_vec.get(4).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                index_type: item_vec.get(5).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                category: item_vec.get(6).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                base_date: item_vec.get(7).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                base_point: item_vec.get(8).and_then(|v| v.as_f64()),
                list_date: item_vec.get(9).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                weight_rule: item_vec.get(10).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                desc: item_vec.get(11).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                exp_date: item_vec.get(12).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
            };
            
            results.push(index_basic);
        }

        Ok(results)
    }
}

/// Database operations
async fn create_index_basic_table(client: &Client) -> Result<()> {
    let create_table_sql = r#"
        CREATE TABLE IF NOT EXISTS index_basic (
            ts_code VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            fullname VARCHAR(200),
            market VARCHAR(20) NOT NULL,
            publisher VARCHAR(100),
            index_type VARCHAR(50),
            category VARCHAR(50),
            base_date VARCHAR(20),
            base_point DOUBLE PRECISION,
            list_date VARCHAR(20),
            weight_rule VARCHAR(100),
            description TEXT,
            exp_date VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_index_basic_market ON index_basic(market);
        CREATE INDEX IF NOT EXISTS idx_index_basic_index_type ON index_basic(index_type);
        CREATE INDEX IF NOT EXISTS idx_index_basic_list_date ON index_basic(list_date);
    "#;

    client
        .batch_execute(create_table_sql)
        .await
        .context("Failed to create index_basic table")?;

    println!("✓ Table 'index_basic' created successfully");
    Ok(())
}

async fn insert_index_data(client: &Client, indices: Vec<IndexBasic>) -> Result<usize> {
    let insert_sql = r#"
        INSERT INTO index_basic (
            ts_code, name, fullname, market, publisher, index_type, 
            category, base_date, base_point, list_date, weight_rule, 
            description, exp_date
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
        )
        ON CONFLICT (ts_code) 
        DO UPDATE SET
            name = EXCLUDED.name,
            fullname = EXCLUDED.fullname,
            market = EXCLUDED.market,
            publisher = EXCLUDED.publisher,
            index_type = EXCLUDED.index_type,
            category = EXCLUDED.category,
            base_date = EXCLUDED.base_date,
            base_point = EXCLUDED.base_point,
            list_date = EXCLUDED.list_date,
            weight_rule = EXCLUDED.weight_rule,
            description = EXCLUDED.description,
            exp_date = EXCLUDED.exp_date,
            updated_at = CURRENT_TIMESTAMP
    "#;

    let mut inserted = 0;
    let mut updated = 0;

    for index in indices {
        let rows = client
            .execute(
                insert_sql,
                &[
                    &index.ts_code,
                    &index.name,
                    &index.fullname,
                    &index.market,
                    &index.publisher,
                    &index.index_type,
                    &index.category,
                    &index.base_date,
                    &index.base_point,
                    &index.list_date,
                    &index.weight_rule,
                    &index.desc,
                    &index.exp_date,
                ],
            )
            .await
            .context("Failed to insert index data")?;

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
    println!("║     Tushare Index Basic Data Ingestion        ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();

    // Get Tushare API token from environment
    let api_token = env::var("TUSHARE_TOKEN")
        .context("TUSHARE_TOKEN environment variable not set. Please set it with your Tushare API token")?;

    // Get database connection string
    let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
        "postgresql://alex:123456@10.0.0.14/stock_data".to_string()
    });

    println!("Connecting to database...");
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
    create_index_basic_table(&client).await?;
    println!();

    // Initialize Tushare client
    let tushare = TushareClient::new(api_token);

    // Fetch all markets
    let markets = vec![
        ("SSE", "上海证券交易所"),
        ("SZSE", "深圳证券交易所"),
        ("CICC", "中证指数公司"),
        ("SW", "申万指数"),
        ("CNI", "国证指数"),
        ("OTH", "其他"),
    ];

    let mut total_inserted = 0;

    for (market_code, market_name) in markets {
        println!("Fetching {} ({}) indices...", market_name, market_code);
        
        match tushare.get_index_basic(Some(market_code)).await {
            Ok(indices) => {
                println!("  ✓ Fetched {} indices", indices.len());
                
                if !indices.is_empty() {
                    println!("  Inserting into database...");
                    match insert_index_data(&client, indices).await {
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
        
        println!();
        
        // Rate limiting - wait 0.5 seconds between requests
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    // Also fetch all indices without market filter
    println!("Fetching all indices...");
    match tushare.get_index_basic(None).await {
        Ok(indices) => {
            println!("  ✓ Fetched {} indices", indices.len());
            
            if !indices.is_empty() {
                println!("  Inserting into database...");
                match insert_index_data(&client, indices).await {
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

    println!();
    println!("╔════════════════════════════════════════════════╗");
    println!("║            Ingestion Complete                  ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("Total records inserted/updated: {}", total_inserted);

    // Query summary
    let count_query = "SELECT market, COUNT(*) as count FROM index_basic GROUP BY market ORDER BY count DESC";
    let rows = client.query(count_query, &[]).await?;
    
    println!();
    println!("Index Summary by Market:");
    println!("─────────────────────────────────────────────────");
    for row in rows {
        let market: String = row.get(0);
        let count: i64 = row.get(1);
        println!("  {:<10} : {} indices", market, count);
    }

    Ok(())
}
