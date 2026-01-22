use chrono::NaiveDate;
use rust_llm_stock::stock_db::get_connection;
use sqlx::Row;
use statrs::statistics::{Data, OrderStatistics, Statistics};
use statrs::statistics::{
    Distribution as _, Max as _, Min as _, OrderStatistics as _, Statistics as _,
};
use std::collections::HashMap; // Add this import for statistics

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;

    println!("=== Updating ML Training Dataset with Global Index & FX Features ===\n");

    let pool = get_connection().await;

    // --- Accept optional start_date from command line ---
    let args: Vec<String> = env::args().collect();
    let user_start_date = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    // Get all unique trade dates from ml_training_dataset, optionally filter by start_date
    let dates: Vec<String> = if let Some(start_date) = user_start_date {
        sqlx::query_scalar(
            "SELECT DISTINCT trade_date FROM ml_training_dataset WHERE trade_date >= $1 ORDER BY trade_date",
        )
        .bind(&start_date)
        .fetch_all(&pool)
        .await?
    } else {
        // If no start_date provided, use max date as default
        let max_date: Option<String> =
            sqlx::query_scalar("SELECT MAX(trade_date) FROM ml_training_dataset")
                .fetch_one(&pool)
                .await?;
        let start_date = max_date.unwrap_or_else(|| "19900101".to_string());
        sqlx::query_scalar(
            "SELECT DISTINCT trade_date FROM ml_training_dataset WHERE trade_date >= $1 ORDER BY trade_date",
        )
        .bind(&start_date)
        .fetch_all(&pool)
        .await?
    };

    println!("Found {} unique trade dates", dates.len());

    let indices = vec!["XIN9", "HSI"];
    let fx_pairs = vec!["USDCNH.FXCM"];

    // --- Load all index and FX data (close, pct_chg) ---
    println!("\nLoading index data...");
    let mut index_data: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for ts_code in indices.iter().chain(fx_pairs.iter()) {
        let rows = sqlx::query(
            "SELECT trade_date, close::DOUBLE PRECISION FROM index_daily WHERE ts_code = $1 ORDER BY trade_date",
        )
        .bind(ts_code)
        .fetch_all(&pool)
        .await?;

        let mut close_data = Vec::new();
        for row in rows {
            let date: String = row.get(0);
            let close: Option<f64> = row.get::<Option<f64>, _>(1);
            if let Some(c) = close {
                close_data.push((date, c));
            }
        }
        println!("  {}: Loaded {} close prices", ts_code, close_data.len());
        index_data.insert(ts_code.to_string(), close_data);
    }

    // --- Calculate features and update database ---
    let mut updated = 0;
    let total_dates = dates.len();
    for (batch_idx, chunk) in dates.chunks(100).enumerate() {
        println!(
            "Processing batch {} ({}-{} of {})",
            batch_idx + 1,
            batch_idx * 100 + 1,
            std::cmp::min((batch_idx + 1) * 100, total_dates),
            total_dates
        );
        for trade_date in chunk {
            // Helper closure to compute pct_chg, vs_ma5, vs_ma20 for a given code
            let compute_features = |code: &str| -> (Option<f64>, Option<f64>, Option<f64>) {
                if let Some(data) = index_data.get(code) {
                    if let Some(pos) = data.iter().position(|(d, _)| d == trade_date) {
                        let close = data[pos].1;
                        let pct_chg = if pos > 0 {
                            Some(((close - data[pos - 1].1) / data[pos - 1].1) * 100.0)
                        } else {
                            Some(0.0)
                        };
                        let ma5 = if pos >= 4 {
                            data[pos - 4..=pos].iter().map(|(_, c)| c).sum::<f64>() / 5.0
                        } else {
                            close
                        };
                        let ma20 = if pos >= 19 {
                            data[pos - 19..=pos].iter().map(|(_, c)| c).sum::<f64>() / 20.0
                        } else {
                            close
                        };
                        let vs_ma5 = Some(((close - ma5) / ma5) * 100.0);
                        let vs_ma20 = Some(((close - ma20) / ma20) * 100.0);
                        (pct_chg, vs_ma5, vs_ma20)
                    } else {
                        (None, None, None)
                    }
                } else {
                    (None, None, None)
                }
            };

            // Helper closure to compute statistics for a given code up to and including trade_date
            let compute_stats = |code: &str| -> (
                Option<f64>,
                Option<f64>,
                Option<f64>,
                Option<f64>,
                Option<f64>,
            ) {
                if let Some(data) = index_data.get(code) {
                    if let Some(pos) = data.iter().position(|(d, _)| d == trade_date) {
                        let closes: Vec<f64> = data[..=pos].iter().map(|(_, c)| *c).collect();
                        if closes.is_empty() {
                            (None, None, None, None, None)
                        } else {
                            let mut closes_data = Data::new(closes.clone());
                            let mean = closes_data.mean();
                            let stddev = closes_data.std_dev();
                            let min = Some(closes_data.min());
                            let max = Some(closes_data.max());
                            let median = Some(closes_data.median());
                            (mean, stddev, min, max, median)
                        }
                    } else {
                        (None, None, None, None, None)
                    }
                } else {
                    (None, None, None, None, None)
                }
            };

            // XIN9 features
            let (xin9_pct_chg, xin9_vs_ma5, xin9_vs_ma20) = compute_features("XIN9");
            let (xin9_mean, xin9_stddev, xin9_min, xin9_max, xin9_median) = compute_stats("XIN9");
            // HSI features
            let (hsi_pct_chg, hsi_vs_ma5, hsi_vs_ma20) = compute_features("HSI");
            let (hsi_mean, hsi_stddev, hsi_min, hsi_max, hsi_median) = compute_stats("HSI");
            // USDCNH features
            let (usdcnh_pct_chg, usdcnh_vs_ma5, usdcnh_vs_ma20) = compute_features("USDCNH.FXCM");
            let (usdcnh_mean, usdcnh_stddev, usdcnh_min, usdcnh_max, usdcnh_median) =
                compute_stats("USDCNH.FXCM");

            sqlx::query(
                "UPDATE ml_training_dataset SET 
                    index_xin9_pct_chg = $1,
                    index_xin9_vs_ma5_pct = $2,
                    index_xin9_vs_ma20_pct = $3,
                    index_xin9_close_mean = $4,
                    index_xin9_close_stddev = $5,
                    index_xin9_close_min = $6,
                    index_xin9_close_max = $7,
                    index_xin9_close_median = $8,
                    index_hsi_pct_chg = $9,
                    index_hsi_vs_ma5_pct = $10,
                    index_hsi_vs_ma20_pct = $11,
                    index_hsi_close_mean = $12,
                    index_hsi_close_stddev = $13,
                    index_hsi_close_min = $14,
                    index_hsi_close_max = $15,
                    index_hsi_close_median = $16,
                    fx_usdcnh_pct_chg = $17,
                    fx_usdcnh_vs_ma5_pct = $18,
                    fx_usdcnh_vs_ma20_pct = $19,
                    fx_usdcnh_close_mean = $20,
                    fx_usdcnh_close_stddev = $21,
                    fx_usdcnh_close_min = $22,
                    fx_usdcnh_close_max = $23,
                    fx_usdcnh_close_median = $24
                 WHERE trade_date = $25",
            )
            .bind(xin9_pct_chg)
            .bind(xin9_vs_ma5)
            .bind(xin9_vs_ma20)
            .bind(xin9_mean)
            .bind(xin9_stddev)
            .bind(xin9_min)
            .bind(xin9_max)
            .bind(xin9_median)
            .bind(hsi_pct_chg)
            .bind(hsi_vs_ma5)
            .bind(hsi_vs_ma20)
            .bind(hsi_mean)
            .bind(hsi_stddev)
            .bind(hsi_min)
            .bind(hsi_max)
            .bind(hsi_median)
            .bind(usdcnh_pct_chg)
            .bind(usdcnh_vs_ma5)
            .bind(usdcnh_vs_ma20)
            .bind(usdcnh_mean)
            .bind(usdcnh_stddev)
            .bind(usdcnh_min)
            .bind(usdcnh_max)
            .bind(usdcnh_median)
            .bind(trade_date)
            .execute(&pool)
            .await?;

            updated += 1;
        }
    }

    println!("\n=== Complete ===");
    println!(
        "Updated {} records with global index and FX features",
        updated
    );

    // Verify
    let stats: (i64, i64, i64, i64) = sqlx::query_as(
        "SELECT 
            COUNT(*) as total,
            COUNT(index_xin9_pct_chg) FILTER (WHERE index_xin9_pct_chg IS NOT NULL) as xin9_nonnull,
            COUNT(index_hsi_pct_chg) FILTER (WHERE index_hsi_pct_chg IS NOT NULL) as hsi_nonnull,
            COUNT(fx_usdcnh_pct_chg) FILTER (WHERE fx_usdcnh_pct_chg IS NOT NULL) as usdcnh_nonnull
         FROM ml_training_dataset",
    )
    .fetch_one(&pool)
    .await?;

    println!("\nVerification:");
    println!("  Total records: {}", stats.0);
    println!(
        "  XIN9 non-null: {} ({:.1}%)",
        stats.1,
        (stats.1 as f64 / stats.0 as f64) * 100.0
    );
    println!(
        "  HSI non-null: {} ({:.1}%)",
        stats.2,
        (stats.2 as f64 / stats.0 as f64) * 100.0
    );
    println!(
        "  USDCNH non-null: {} ({:.1}%)",
        stats.3,
        (stats.3 as f64 / stats.0 as f64) * 100.0
    );

    Ok(())
}
