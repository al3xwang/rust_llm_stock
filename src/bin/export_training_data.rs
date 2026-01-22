// Export ML training data from ml_training_dataset to CSV, omitting target columns from features.
// Only use Option<f64> safe arithmetic and handle missing data gracefully.

use sqlx::{Column, Row, ValueRef, postgres::PgPoolOptions};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load DB URL from env or use default
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:12341234@localhost:5432/research".to_string());
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&db_url)
        .await?;

    // Query: select stocks listed at least 7 years ago, with 2 years warmup and 5 years modeling data, filter out pe/pe_ttm/dv_ttm nulls
    // Use a CTE to select 1000 random eligible stocks
    let rows = sqlx::query(
        r#"
        WITH eligible_stocks AS (
            SELECT s.ts_code
            FROM stock_basic s
            WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '7 years'), 'YYYYMMDD')
              AND EXISTS (
                  SELECT 1 FROM ml_training_dataset d
                  WHERE d.ts_code = s.ts_code
                    AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '7 years'), 'YYYYMMDD')
                    AND d.pe IS NOT NULL AND d.pe_ttm IS NOT NULL AND d.dv_ttm IS NOT NULL
              )
            ORDER BY RANDOM()
            LIMIT 1000
        )
        SELECT d.*
        FROM ml_training_dataset d
        JOIN eligible_stocks e ON d.ts_code = e.ts_code
        WHERE d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '7 years'), 'YYYYMMDD')
          AND d.pe IS NOT NULL AND d.pe_ttm IS NOT NULL AND d.dv_ttm IS NOT NULL
        ORDER BY d.ts_code, d.trade_date
        "#,
    )
    .fetch_all(&pool)
    .await?;

    if rows.is_empty() {
        println!("No data found in ml_training_dataset.");
        return Ok(());
    }

    // Get column names
    let columns = rows[0].columns();
    let all_colnames: Vec<&str> = columns.iter().map(|c| c.name()).collect();

    // Define which columns are targets/labels (do NOT export as features)
    let target_cols = [
        "next_day_return",
        "next_day_direction",
        "next_3day_return",
        "next_3day_direction",
    ];

    // Always exclude these from features (metadata)
    let meta_cols: [&str; 2] = ["id", "created_at"];

    // Always include these as identifiers in CSV
    let id_cols = ["ts_code", "trade_date"];

    // Build feature columns: all except targets and meta, but keep id_cols at front
    let feature_cols: Vec<&str> = all_colnames
        .iter()
        .filter(|c| !target_cols.contains(c) && !meta_cols.contains(c) && !id_cols.contains(c))
        .copied()
        .collect();

    // Final CSV columns: id_cols + feature_cols + target_cols
    let mut csv_cols = Vec::new();
    csv_cols.extend_from_slice(&id_cols);
    csv_cols.extend_from_slice(&feature_cols);
    csv_cols.extend_from_slice(&target_cols);

    // Write CSV
    let file = File::create("training_data.csv")?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "{}", csv_cols.join(","))?;

    for row in &rows {
        let mut vals = Vec::with_capacity(csv_cols.len());
        for &col in &csv_cols {
            let idx = all_colnames.iter().position(|&c| c == col).unwrap();
            let val = row.try_get_raw(idx);
            let s = if val.is_ok() && !val.as_ref().unwrap().is_null() {
                // Try as string, f64, i32, i16, bool
                if let Ok(v) = row.try_get::<&str, _>(idx) {
                    v.to_string()
                } else if let Ok(v) = row.try_get::<f64, _>(idx) {
                    format!("{:.6}", v)
                } else if let Ok(v) = row.try_get::<i32, _>(idx) {
                    v.to_string()
                } else if let Ok(v) = row.try_get::<i16, _>(idx) {
                    v.to_string()
                } else if let Ok(v) = row.try_get::<bool, _>(idx) {
                    v.to_string()
                } else {
                    "".to_string()
                }
            } else {
                "".to_string()
            };
            vals.push(s);
        }
        writeln!(writer, "{}", vals.join(","))?;
    }

    println!(
        "Exported {} rows to training_data.csv with {} columns.",
        rows.len(),
        csv_cols.len()
    );

    // Group rows by ts_code
    use std::collections::HashMap;
    let mut stock_map: HashMap<String, Vec<&sqlx::postgres::PgRow>> = HashMap::new();
    for row in &rows {
        let ts_code: String = row.try_get("ts_code").unwrap_or_default();
        stock_map.entry(ts_code).or_default().push(row);
    }

    // For each stock, split last 5 years as train/val/test, use first 2 years as warmup only
    let mut train_rows = Vec::new();
    let mut val_rows = Vec::new();
    let mut test_rows = Vec::new();
    for (_ts_code, stock_rows) in stock_map {
        // Sort by trade_date ascending
        let mut sorted = stock_rows;
        sorted.sort_by_key(|row| row.try_get::<String, _>("trade_date").unwrap_or_default());
        let total = sorted.len();
        // Lower minimum: require at least 2 years warmup + 1 year modeling (approx 660 days)
        let warmup = (2.0_f32 * 240.0_f32).round() as usize; // 2 years warmup (approx)
        let min_model = (1.0_f32 * 220.0_f32).round() as usize; // 1 year modeling (approx)
        if total < warmup + min_model {
            continue;
        }
        let model_data = &sorted[warmup..]; // after warmup
        let n = model_data.len();
        if n < 10 {
            continue;
        }
        let test_size = (n as f32 * 0.1).round() as usize;
        let val_size = (n as f32 * 0.2).round() as usize;
        let train_size = n - val_size - test_size;
        train_rows.extend_from_slice(&model_data[..train_size]);
        val_rows.extend_from_slice(&model_data[train_size..train_size + val_size]);
        test_rows.extend_from_slice(&model_data[train_size + val_size..]);
    }

    // Write CSVs for train/val/test
    let mut write_csv =
        |fname: &str, rows: &Vec<&sqlx::postgres::PgRow>| -> Result<(), Box<dyn Error>> {
            let file = File::create(fname)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", csv_cols.join(","))?;
            for row in rows {
                let mut vals = Vec::with_capacity(csv_cols.len());
                for &col in &csv_cols {
                    let idx = all_colnames.iter().position(|&c| c == col).unwrap();
                    let val = row.try_get_raw(idx);
                    let s = if val.is_ok() && !val.as_ref().unwrap().is_null() {
                        if let Ok(v) = row.try_get::<&str, _>(idx) {
                            v.to_string()
                        } else if let Ok(v) = row.try_get::<f64, _>(idx) {
                            format!("{:.6}", v)
                        } else if let Ok(v) = row.try_get::<i32, _>(idx) {
                            v.to_string()
                        } else if let Ok(v) = row.try_get::<i16, _>(idx) {
                            v.to_string()
                        } else if let Ok(v) = row.try_get::<bool, _>(idx) {
                            v.to_string()
                        } else {
                            "".to_string()
                        }
                    } else {
                        "".to_string()
                    };
                    vals.push(s);
                }
                writeln!(writer, "{}", vals.join(","))?;
            }
            Ok(())
        };
    write_csv("train.csv", &train_rows)?;
    write_csv("val.csv", &val_rows)?;
    write_csv("test.csv", &test_rows)?;
    println!(
        "Exported train.csv: {} rows, val.csv: {}, test.csv: {}",
        train_rows.len(),
        val_rows.len(),
        test_rows.len()
    );

    Ok(())
}
