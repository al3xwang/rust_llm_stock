// Export ML training data from ml_training_dataset to CSV, omitting target columns from features.
// Only use Option<f64> safe arithmetic and handle missing data gracefully.

use sqlx::{Column, Row, ValueRef, postgres::PgPoolOptions};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "export_training_data")]
struct Cli {
    /// Minimum number of years of history required per stock to include in export (default: 2)
    #[clap(long, default_value_t = 2usize)]
    min_years: usize,

    /// Minimum total amount traded in the last ~5 days (CN¥). Stocks with less are excluded (default: 0 -> disabled)
    #[clap(long, default_value_t = 0i64)]
    min_5day_amount: i64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Load DB URL from env or use default
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:12341234@localhost:5432/research".to_string());
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&db_url)
        .await?;

    // Query: select stocks with data covering the most recent 5 years (we will export latest 5 years per stock)
    // Selection refinement: compute average traded amount over the last 3 months and select the top 30% by liquidity.
    // Always include stocks whose code starts with the desired prefixes regardless of liquidity (to ensure coverage).
    // Build the SQL selection; if a minimum 5-day amount threshold is configured, apply it in the selection
    let min_5day_amount = cli.min_5day_amount;
    let min_years_cond = if cli.min_years > 0 {
        format!("AND EXISTS (SELECT 1 FROM ml_training_dataset d WHERE d.ts_code = s.ts_code AND d.trade_date <= TO_CHAR((CURRENT_DATE - INTERVAL '{} years'), 'YYYYMMDD'))", cli.min_years)
    } else {
        "".to_string()
    };
    let rows = if min_5day_amount > 0 {
        sqlx::query(&format!(
            r#"
                WITH eligible AS (
                    -- Stocks with sufficient listing history and presence in ml_training_dataset
                    -- Exclude names starting with ST or *ST and stocks whose latest close < 2 CNY
                    SELECT s.ts_code
                    FROM stock_basic s
                    JOIN LATERAL (
                        SELECT a.close
                        FROM adjusted_stock_daily a
                        WHERE a.ts_code = s.ts_code
                        ORDER BY a.trade_date DESC
                        LIMIT 1
                    ) lc ON lc.close >= 2.0
                    WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                      AND NOT (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
                      {min_years_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                      )
                ),
                recent_liq AS (
                    -- Explicitly compute avg(amount) over the last 3 months per stock and expose count of days
                    SELECT a.ts_code,
                           AVG(a.amount)::float8 AS avg_amount_3m,
                           COUNT(*) AS n_days_3m
                    FROM adjusted_stock_daily a
                    WHERE a.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '3 months'), 'YYYYMMDD')
                    GROUP BY a.ts_code
                ),
                last_5d AS (
                    -- Sum traded amount over the last ~5 trading days (approx using 7 calendar days window)
                    SELECT a.ts_code,
                           COALESCE(SUM(a.amount), 0.0)::float8 AS amount_5d
                    FROM adjusted_stock_daily a
                    WHERE a.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '7 days'), 'YYYYMMDD')
                    GROUP BY a.ts_code
                ),
                liq_rank AS (
                    SELECT r.ts_code, COALESCE(r.avg_amount_3m, 0.0) AS avg_amount_3m,
                           NTILE(100) OVER (ORDER BY COALESCE(r.avg_amount_3m, 0.0) DESC) AS pct_rank
                    FROM recent_liq r
                    JOIN eligible e ON e.ts_code = r.ts_code
                ),
                selected_stocks AS (
                    -- Top 30 percentile by 3-month traded amount AND meeting last-5d amount threshold
                    SELECT r.ts_code FROM liq_rank r JOIN last_5d l ON l.ts_code = r.ts_code WHERE pct_rank <= 30 AND l.amount_5d >= {min_amt}
                    UNION
                    -- Randomly include half of prefix '60' and '00' stocks; keep full inclusion for 30/68/9
                    SELECT ts_code FROM (
                        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY random()) AS rn
                        FROM eligible WHERE ts_code LIKE '60%'
                    ) t WHERE rn <= CEIL((SELECT COUNT(*)::float FROM eligible WHERE ts_code LIKE '60%') / 2.0)
                    UNION
                    SELECT ts_code FROM (
                        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY random()) AS rn
                        FROM eligible WHERE ts_code LIKE '00%'
                    ) t WHERE rn <= CEIL((SELECT COUNT(*)::float FROM eligible WHERE ts_code LIKE '00%') / 2.0)
                    UNION
                    SELECT ts_code FROM eligible WHERE ts_code LIKE '30%' OR ts_code LIKE '68%' OR ts_code LIKE '9%'
                )
                SELECT d.*
                FROM ml_training_dataset d
                JOIN selected_stocks s ON d.ts_code = s.ts_code
                WHERE d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                ORDER BY d.ts_code, d.trade_date
        "#,
            min_amt = min_5day_amount,
            min_years_cond = min_years_cond
        ))
        .fetch_all(&pool)
        .await?
    } else {
        sqlx::query(&format!(
            r#"
                WITH eligible AS (
                    -- Stocks with sufficient listing history and presence in ml_training_dataset
                    -- Exclude names starting with ST or *ST and stocks whose latest close < 2 CNY
                    SELECT s.ts_code
                    FROM stock_basic s
                    JOIN LATERAL (
                        SELECT a.close
                        FROM adjusted_stock_daily a
                        WHERE a.ts_code = s.ts_code
                        ORDER BY a.trade_date DESC
                        LIMIT 1
                    ) lc ON lc.close >= 2.0
                    WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                      AND NOT (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
                      {min_years_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                      )
                ),
                recent_liq AS (
                    -- Explicitly compute avg(amount) over the last 3 months per stock and expose count of days
                    SELECT a.ts_code,
                           AVG(a.amount)::float8 AS avg_amount_3m,
                           COUNT(*) AS n_days_3m
                    FROM adjusted_stock_daily a
                    WHERE a.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '3 months'), 'YYYYMMDD')
                    GROUP BY a.ts_code
                ),
                liq_rank AS (
                    SELECT r.ts_code, COALESCE(r.avg_amount_3m, 0.0) AS avg_amount_3m,
                           NTILE(100) OVER (ORDER BY COALESCE(r.avg_amount_3m, 0.0) DESC) AS pct_rank
                    FROM recent_liq r
                    JOIN eligible e ON e.ts_code = r.ts_code
                ),
                selected_stocks AS (
                    -- Top 30 percentile by 3-month traded amount OR explicit prefix inclusion
                    SELECT r.ts_code FROM liq_rank r WHERE pct_rank <= 30
                    UNION
                    -- Randomly include half of prefix '60' and '00' stocks; keep full inclusion for 30/68/9
                    SELECT ts_code FROM (
                        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY random()) AS rn
                        FROM eligible WHERE ts_code LIKE '60%'
                    ) t WHERE rn <= CEIL((SELECT COUNT(*)::float FROM eligible WHERE ts_code LIKE '60%') / 2.0)
                    UNION
                    SELECT ts_code FROM (
                        SELECT ts_code, ROW_NUMBER() OVER (ORDER BY random()) AS rn
                        FROM eligible WHERE ts_code LIKE '00%'
                    ) t WHERE rn <= CEIL((SELECT COUNT(*)::float FROM eligible WHERE ts_code LIKE '00%') / 2.0)
                    UNION
                    SELECT ts_code FROM eligible WHERE ts_code LIKE '30%' OR ts_code LIKE '68%' OR ts_code LIKE '9%'
                )
                SELECT d.*
                FROM ml_training_dataset d
                JOIN selected_stocks s ON d.ts_code = s.ts_code
                WHERE d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
                ORDER BY d.ts_code, d.trade_date
        "#, min_years_cond = min_years_cond))
        .fetch_all(&pool)
        .await?
    };


    if rows.is_empty() {
        println!("No data found in ml_training_dataset.");
        return Ok(());
    }

    // Diagnostics: counts of candidate stocks and exclusions by price/name
    // Base candidate universe: stocks listed long enough and with ml_training_dataset rows in last 5 years
    let total_candidates: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          )
        "#,
    )
    .fetch_one(&pool)
    .await?;

    let excluded_by_price: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        JOIN LATERAL (
            SELECT a.close FROM adjusted_stock_daily a WHERE a.ts_code = s.ts_code ORDER BY a.trade_date DESC LIMIT 1
        ) lc ON lc.close < 2.0
        WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          )
        "#,
    )
    .fetch_one(&pool)
    .await?;

    let excluded_by_name: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
          AND s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          )
        "#,
    )
    .fetch_one(&pool)
    .await?;

    let excluded_by_both: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        JOIN LATERAL (
            SELECT a.close FROM adjusted_stock_daily a WHERE a.ts_code = s.ts_code ORDER BY a.trade_date DESC LIMIT 1
        ) lc ON lc.close < 2.0
        WHERE (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
          AND s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          )
        "#,
    )
    .fetch_one(&pool)
    .await?;

    let excluded_by_short_history: i64 = if cli.min_years > 0 {
        sqlx::query_scalar(&format!(r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE s.list_date <= TO_CHAR((CURRENT_DATE - INTERVAL '5 years'), 'YYYYMMDD')
          AND NOT EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date <= TO_CHAR((CURRENT_DATE - INTERVAL '{} years'), 'YYYYMMDD')
          )
        "#, cli.min_years))
        .fetch_one(&pool)
        .await?
    } else {
        0
    };

    println!("Candidate stocks before filters: {}", total_candidates);
    println!("  Excluded by latest close < 2.0: {}", excluded_by_price);
    println!("  Excluded by name (ST/*ST): {}", excluded_by_name);
    println!("  Excluded by both price & name: {}", excluded_by_both);
    if cli.min_years > 0 {
        println!("  Excluded by short history (< {} years): {}", cli.min_years, excluded_by_short_history);
    }

    // Compute selected stock set and print basic coverage statistics
    use std::collections::HashSet;
    let mut sel_set: HashSet<String> = HashSet::new();
    for row in &rows {
        let ts_code: String = row.try_get("ts_code").unwrap_or_default();
        sel_set.insert(ts_code);
    }
    println!("Selected stock codes: {}", sel_set.len());
    // Prefix coverage (60,00,30,68,9)
    let prefixes = ["60", "00", "30", "68", "9"];
    for p in &prefixes {
        let cnt = sel_set.iter().filter(|s| s.starts_with(p)).count();
        println!("  Prefix {}: {} stocks", p, cnt);
    }

    // Get column names (normalize by trimming whitespace)
    let columns = rows[0].columns();
    let all_colnames: Vec<String> = columns
        .iter()
        .map(|c| c.name().trim().to_string())
        .collect();

    // Target columns (lowercase names used for comparison)
    let target_cols = ["next_day_return", "next_day_direction", "next_3day_return", "next_3day_direction"];

    // Always exclude these from features (metadata)
    let meta_cols: [&str; 2] = ["id", "created_at"];


    // Always include these as identifiers in CSV
    let id_cols = ["ts_code", "trade_date"];

    // Build feature columns: all except targets and meta, but keep id_cols at front
    // Use normalized lowercase comparisons to avoid whitespace/case mismatches.
    let feature_cols: Vec<String> = all_colnames
        .iter()
        .filter(|c| {
            let lc = c.to_lowercase();
            // strictly exclude any column that is a known target or meta or id
            if target_cols.iter().any(|t| lc == *t) {
                return false;
            }
            if meta_cols.iter().any(|m| lc == *m) {
                return false;
            }
            if id_cols.iter().any(|i| lc == *i) {
                return false;
            }
            // Also exclude any column that appears to be a future-derived field (starts with "next_")
            if lc.starts_with("next_") {
                return false;
            }
            true
        })
        .cloned()
        .collect();

    // Final CSV columns: id_cols + feature_cols + target_cols
    let mut csv_cols: Vec<String> = Vec::new();
    for &id in &id_cols {
        csv_cols.push(id.to_string());
    }
    csv_cols.extend_from_slice(&feature_cols);
    for &t in &target_cols {
        csv_cols.push(t.to_string());
    }

    // Sanity checks: ensure no feature equals any target (case-insensitive)
    for f in &feature_cols {
        for &t in &target_cols {
            if f.eq_ignore_ascii_case(t) {
                panic!("Feature column '{}' matches target column '{}' — aborting export", f, t);
            }
        }
    }

    // Skipping full training_data.csv export — this file is not used for model training.
    println!("Skipping export of training_data.csv (disabled)");

    println!("CSV columns: {}", csv_cols.len());

    // Group rows by ts_code
    use std::collections::HashMap;
    let mut stock_map: HashMap<String, Vec<&sqlx::postgres::PgRow>> = HashMap::new();
    for row in &rows {
        let ts_code: String = row.try_get("ts_code").unwrap_or_default();
        stock_map.entry(ts_code).or_default().push(row);
    }

    // For each stock, take the most recent 5 years of data and split into: 3.5y train | 1y val | 0.5y test.
    // For stocks with less than 5 years, split available days proportionally (70% train, 20% val, 10% test).
    let mut train_rows = Vec::new();
    let mut val_rows = Vec::new();
    let mut test_rows = Vec::new();
    // Trading days approximated as 240 per year
    let trading_days_year = 240usize;
    let train_days = (3.5_f32 * trading_days_year as f32).round() as usize; // 3.5 years
    let val_days = 1 * trading_days_year; // 1 year
    let test_days = (trading_days_year as f32 * 0.5).round() as usize; // 0.5 year
    let required_per_stock = train_days + val_days + test_days; // 5 years total (~1200 days)
    let mut full_coverage_stocks = 0usize;
    let mut partial_coverage_stocks = 0usize;
    let mut skipped_by_min_years = 0usize;
    // Diagnostics for date-based splits: how many stocks lack rows in each split
    let mut missing_train = 0usize;
    let mut missing_val = 0usize;
    let mut missing_test = 0usize;
    // min_years from CLI: convert to a minimum required trading days threshold when > 0
    let min_required_days = if cli.min_years > 0 {
        (cli.min_years * trading_days_year) as usize
    } else {
        0usize
    };

    for (ts_code, stock_rows) in stock_map {
        // Sort by trade_date ascending
        let mut sorted = stock_rows;
        sorted.sort_by_key(|row| row.try_get::<String, _>("trade_date").unwrap_or_default());
        let total = sorted.len();

        // If user requested a minimum years threshold, skip stocks that don't meet it
        if min_required_days > 0 && total < min_required_days {
            skipped_by_min_years += 1;
            eprintln!("⚠️  Skipping {}: only {} days history (< {} days for {} years)", ts_code, total, min_required_days, cli.min_years);
            continue;
        }

        // Split by fixed date cutoffs (preferred):
        // Train: trade_date <= 20250229
        // Val:   20250301 <= trade_date <= 20251103
        // Test:  trade_date >= 20251104
        let model_data = &sorted[..];
        let mut train_part: Vec<&sqlx::postgres::PgRow> = Vec::new();
        let mut val_part: Vec<&sqlx::postgres::PgRow> = Vec::new();
        let mut test_part: Vec<&sqlx::postgres::PgRow> = Vec::new();
        let train_cutoff = "20250229";
        let val_cutoff = "20251103";
        for row in model_data.iter() {
            let d: String = row.try_get("trade_date").unwrap_or_default();
            if d.as_str() <= train_cutoff {
                train_part.push(row);
            } else if d.as_str() <= val_cutoff {
                val_part.push(row);
            } else {
                test_part.push(row);
            }
        }

        // Track missing bins for diagnostics
        if train_part.is_empty() {
            missing_train += 1;
        }
        if val_part.is_empty() {
            missing_val += 1;
        }
        if test_part.is_empty() {
            missing_test += 1;
        }

        // If no rows fall into any of the bins (very sparse history), fall back to proportional split
        if train_part.is_empty() && val_part.is_empty() && test_part.is_empty() {
            let n = model_data.len();
            if n < 3 {
                train_rows.extend_from_slice(model_data);
                partial_coverage_stocks += 1;
                continue;
            }
            let test_size = (n as f32 * 0.10).round() as usize;
            let val_size = (n as f32 * 0.20).round() as usize;
            let train_size = n.saturating_sub(val_size + test_size);
            if train_size > 0 {
                train_rows.extend_from_slice(&model_data[..train_size]);
            }
            if val_size > 0 {
                val_rows.extend_from_slice(&model_data[train_size..train_size + val_size]);
            }
            if test_size > 0 {
                test_rows.extend_from_slice(&model_data[train_size + val_size..]);
            }
            partial_coverage_stocks += 1;
            continue;
        }

        // Use date-split parts
        if !train_part.is_empty() {
            train_rows.extend_from_slice(&train_part);
        }
        if !val_part.is_empty() {
            val_rows.extend_from_slice(&val_part);
        }
        if !test_part.is_empty() {
            test_rows.extend_from_slice(&test_part);
        }

        // Track coverage: if a stock has at least required_per_stock days across the window, consider full_coverage
        if total >= required_per_stock {
            full_coverage_stocks += 1;
        } else {
            partial_coverage_stocks += 1;
        }
    }

    println!("Stocks with full 5y coverage: {}, partial: {}, skipped by --min-years: {}, per-stock target days: {} (train {} val {} test {}), date cutoffs: train<=20250229 val<=20251103 test>20251103", full_coverage_stocks, partial_coverage_stocks, skipped_by_min_years, required_per_stock, train_days, val_days, test_days);
    println!("Stocks missing bins by date-split: missing_train: {}, missing_val: {}, missing_test: {}", missing_train, missing_val, missing_test);

    // Write CSVs for train/val/test
    let mut write_csv =
        |fname: &str, rows: &Vec<&sqlx::postgres::PgRow>| -> Result<(), Box<dyn Error>> {
            let file = File::create(fname)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", csv_cols.join(","))?;
            for row in rows {
                let mut vals = Vec::with_capacity(csv_cols.len());
                    for col in &csv_cols {
                        let idx = all_colnames.iter().position(|c| c == col).unwrap();
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
