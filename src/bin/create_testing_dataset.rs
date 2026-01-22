use std::error::Error;

use rust_llm_stock::stock_db::get_connection;
use sqlx::{Pool, Postgres, Row};

async fn create_testing_dataset_table(pool: &Pool<Postgres>) -> Result<(), Box<dyn Error>> {
    // Drop existing table
    sqlx::query("DROP TABLE IF EXISTS testing_dataset CASCADE")
        .execute(pool)
        .await?;

    // Create new table
    sqlx::query(
        r#"
        CREATE TABLE testing_dataset (
            ts_code VARCHAR(20) NOT NULL,
            trade_date VARCHAR(8) NOT NULL,
            open NUMERIC(10, 2),
            high NUMERIC(10, 2),
            low NUMERIC(10, 2),
            close NUMERIC(10, 2),
            pre_close NUMERIC(10, 2),
            volume NUMERIC(20, 2),
            amount NUMERIC(20, 4),
            adj_factor NUMERIC(10, 4),
            PRIMARY KEY (ts_code, trade_date)
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Create indexes
    sqlx::query("CREATE INDEX idx_testing_dataset_ts_code ON testing_dataset(ts_code)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX idx_testing_dataset_trade_date ON testing_dataset(trade_date)")
        .execute(pool)
        .await?;

    println!("✓ Created testing_dataset table");
    Ok(())
}

async fn process_stock(
    pool: &Pool<Postgres>,
    ts_code: &str,
    idx: usize,
    total: usize,
) -> Result<usize, Box<dyn Error>> {
    // Fetch all data for the stock, ordered from newest to oldest for backward adjustment
    let rows = sqlx::query(
        r#"
        SELECT d.ts_code, d.trade_date, 
               CAST(d.open AS TEXT) as open, 
               CAST(d.high AS TEXT) as high, 
               CAST(d.low AS TEXT) as low, 
               CAST(d.close AS TEXT) as close, 
               CAST(d.pre_close AS TEXT) as pre_close, 
               CAST(d.vol AS TEXT) as volume, 
               CAST(d.amount AS TEXT) as amount, 
               CAST(COALESCE(a.adj_factor, 1.0) AS TEXT) as adj_factor
        FROM stock_daily d
        LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        WHERE d.ts_code = $1
        ORDER BY d.trade_date DESC
        "#,
    )
    .bind(ts_code)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        println!("[{}/{}] - {}: skipped (no data)", idx + 1, total, ts_code);
        return Ok(0);
    }

    let mut tx = pool.begin().await?;
    let mut inserted_count = 0;

    for row in rows.iter() {
        let ts_code: String = row.get("ts_code");
        let trade_date: String = row.get("trade_date");
        let open: Option<String> = row.try_get("open").ok();
        let high: Option<String> = row.try_get("high").ok();
        let low: Option<String> = row.try_get("low").ok();
        let close: Option<String> = row.try_get("close").ok();
        let pre_close: Option<String> = row.try_get("pre_close").ok();
        let volume: Option<String> = row.try_get("volume").ok();
        let amount: Option<String> = row.try_get("amount").ok();
        let adj_factor: String = row.get("adj_factor");

        // Skip if essential prices are missing
        if open.is_none() || high.is_none() || low.is_none() || close.is_none() {
            continue;
        }

        // Insert raw prices (no adjustment)
        sqlx::query(
            r#"
            INSERT INTO testing_dataset (
                ts_code, trade_date, open, high, low, close, pre_close, 
                volume, amount, adj_factor
            ) VALUES (
                $1, $2, $3::NUMERIC, $4::NUMERIC, $5::NUMERIC, $6::NUMERIC, $7::NUMERIC, $8::NUMERIC, $9::NUMERIC, $10::NUMERIC
            )
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                pre_close = EXCLUDED.pre_close,
                volume = EXCLUDED.volume,
                amount = EXCLUDED.amount,
                adj_factor = EXCLUDED.adj_factor
            "#
        )
        .bind(&ts_code)
        .bind(&trade_date)
        .bind(open)
        .bind(high)
        .bind(low)
        .bind(close)
        .bind(pre_close)
        .bind(volume)
        .bind(amount)
        .bind(adj_factor)
        .execute(&mut *tx)
        .await?;

        inserted_count += 1;
    }

    tx.commit().await?;

    println!(
        "[{}/{}] ✓ {}: {} records",
        idx + 1,
        total,
        ts_code,
        inserted_count
    );

    Ok(inserted_count)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let pool = get_connection().await;

    // Create the testing_dataset table
    create_testing_dataset_table(&pool).await?;

    // Get all unique stocks from adjusted_stock_daily
    let stocks = sqlx::query_as::<_, (String,)>(
        r#"
        SELECT DISTINCT ts_code 
        FROM adjusted_stock_daily 
        ORDER BY ts_code
        "#,
    )
    .fetch_all(&pool)
    .await?;

    println!("\n🚀 Processing {} stocks...\n", stocks.len());

    let start_time = std::time::Instant::now();
    let mut total_records = 0;
    let mut successful_stocks = 0;

    for (idx, (ts_code,)) in stocks.iter().enumerate() {
        match process_stock(&pool, ts_code, idx, stocks.len()).await {
            Ok(count) => {
                total_records += count;
                if count > 0 {
                    successful_stocks += 1;
                }
            }
            Err(e) => {
                eprintln!("[{}/{}] ✗ {}: {}", idx + 1, stocks.len(), ts_code, e);
            }
        }
    }

    let elapsed = start_time.elapsed();

    println!("\n📊 Summary:");
    println!("  - Processed: {} stocks", stocks.len());
    println!("  - Successful: {} stocks", successful_stocks);
    println!("  - Total records: {}", total_records);
    println!("  - Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  - Average: {:.2}s per stock",
        elapsed.as_secs_f64() / stocks.len() as f64
    );

    Ok(())
}
