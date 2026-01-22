use anyhow::Result;
use serde::Deserialize;
use crate::db::DbClient;
use chrono::Datelike;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct CustomQuote {
    pub symbol: String,
    pub timestamp: i64, // Unix timestamp
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

#[allow(dead_code)]
pub async fn fetch_and_save_custom_data(db: &DbClient, url: &str) -> Result<()> {
    println!("Fetching data from custom source: {}", url);
    
    // Placeholder for actual HTTP request
    // let response = reqwest::get(url).await?.json::<Vec<CustomQuote>>().await?;
    
    // For demonstration, let's generate some dummy data if the URL is "dummy"
    let quotes = if url == "dummy" {
        vec![
            CustomQuote {
                symbol: "AAPL".to_string(),
                timestamp: 1700000000,
                open: 150.0,
                high: 155.0,
                low: 149.0,
                close: 153.0,
                volume: 1000000.0,
            },
            CustomQuote {
                symbol: "AAPL".to_string(),
                timestamp: 1700086400,
                open: 153.0,
                high: 158.0,
                low: 152.0,
                close: 157.0,
                volume: 1200000.0,
            },
        ]
    } else {
        // In a real scenario:
        // reqwest::get(url).await?.json::<Vec<CustomQuote>>().await?
        vec![]
    };

    println!("Fetched {} quotes. Saving to database...", quotes.len());

    for quote in quotes {
        // Convert timestamp to DateTime to extract month and weekday
        let datetime = chrono::DateTime::from_timestamp(quote.timestamp, 0)
            .map(|dt| dt.naive_utc())
            .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap().naive_utc());
        let month = datetime.month() as i32; // 1-12
        let weekday = datetime.weekday().num_days_from_monday() as i32; // 0=Monday, 6=Sunday
        
        // For now, use placeholder values for indicators (would need full history to calculate properly)
        let sma5 = quote.close;
        let sma20 = quote.close;
        let rsi = 50.0;
        let daily_return = 0.0;
        let volume_ratio = 1.0;
        let quarter = match month {
            1..=3 => 1,
            4..=6 => 2,
            7..=9 => 3,
            _ => 4,
        };
        
        db.save_quote(
            &quote.symbol,
            quote.timestamp,
            quote.open,
            quote.high,
            quote.low,
            quote.close,
            quote.volume,
            month,
            weekday,
            sma5,
            sma20,
            rsi,
            daily_return,
            volume_ratio,
            quarter,
        ).await?;
    }
    
    println!("Data saved successfully.");
    Ok(())
}
