use sqlx::{Pool, Postgres};
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    dotenv::dotenv().ok();
    let database_url = env::var("DATABASE_URL")?;
    let pool = Pool::<Postgres>::connect(&database_url).await?;

    println!("\n=== Checking NULL values in ml_training_dataset ===\n");
    
    // Get total record count
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ml_training_dataset")
        .fetch_one(&pool)
        .await?;
    
    println!("Total records: {}\n", total);
    
    // Check each column for nulls
    let columns = vec![
        // Categorical features
        "industry", "act_ent_type", 
        // OHLCV
        "open", "high", "low", "close", "pre_close", "volume", "amount", "adj_factor",
        // Time features
        "month", "weekday", "quarter", "week_no",
        // Price changes
        "change", "pct_change",
        // EMAs
        "ema_5", "ema_10", "ema_20", "ema_30", "ema_60",
        // SMAs
        "sma_5", "sma_10", "sma_20",
        // MACD
        "macd_line", "macd_signal", "macd_histogram",
        "macd_weekly_line", "macd_weekly_signal",
        "macd_monthly_line", "macd_monthly_signal",
        // RSI
        "rsi_14",
        // KDJ
        "kdj_k", "kdj_d", "kdj_j",
        // Bollinger Bands
        "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_percent_b",
        // Volatility
        "atr", "volatility_5", "volatility_20",
        // Volume indicators
        "asi", "obv", "volume_ratio",
        // Momentum
        "price_momentum_5", "price_momentum_10", "price_momentum_20",
        // Price position
        "price_position_52w",
        // Candlestick
        "body_size", "upper_shadow", "lower_shadow",
        // Trend
        "trend_strength",
        // Target
        "next_day_return", "next_day_direction",
    ];
    
    let mut null_columns = Vec::new();
    
    for col in &columns {
        let query = format!("SELECT COUNT(*) - COUNT({}) as null_count FROM ml_training_dataset", col);
        let null_count: i64 = sqlx::query_scalar(&query)
            .fetch_one(&pool)
            .await?;
        
        if null_count > 0 {
            let pct = (null_count as f64 / total as f64) * 100.0;
            null_columns.push((col, null_count, pct));
        }
    }
    
    if null_columns.is_empty() {
        println!("✓ All columns are fully populated with non-null values!");
    } else {
        println!("Columns with NULL values:\n");
        println!("{:<25} {:>12} {:>10}", "Column", "NULL Count", "NULL %");
        println!("{:-<48}", "");
        
        null_columns.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (col, count, pct) in null_columns {
            println!("{:<25} {:>12} {:>9.2}%", col, count, pct);
        }
    }
    
    Ok(())
}
