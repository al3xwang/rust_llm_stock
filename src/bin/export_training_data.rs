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

    /// Train cutoff date (inclusive) in YYYYMMDD (default: 20250229)
    #[clap(long, default_value = "20250229")]
    train_cutoff: String,

    /// Validation window(s) as START:END in YYYYMMDD format. Repeatable. (default: 20250301:20251103)
    #[clap(long = "val-window")]
    val_windows: Vec<String>,
}

pub fn parse_val_windows(cli_val_windows: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut val_windows_parsed: Vec<(String, String)> = Vec::new();
    if cli_val_windows.is_empty() {
        val_windows_parsed.push(("20250301".to_string(), "20251103".to_string()));
        return Ok(val_windows_parsed);
    }
    for vw in cli_val_windows {
        if !vw.contains(':') {
            return Err(format!("Invalid --val-window '{}', expected START:END", vw));
        }
        let parts: Vec<&str> = vw.splitn(2, ':').collect();
        let start = parts[0].to_string();
        let end = parts[1].to_string();
        if start > end {
            return Err(format!("Invalid val-window {}: start > end", vw));
        }
        val_windows_parsed.push((start, end));
    }
    val_windows_parsed.sort_by(|a, b| a.0.cmp(&b.0));
    for i in 1..val_windows_parsed.len() {
        if val_windows_parsed[i].0 <= val_windows_parsed[i - 1].1 {
            return Err(format!("Validation windows overlap or are contiguous: {} <= {}", val_windows_parsed[i].0, val_windows_parsed[i - 1].1));
        }
    }
    Ok(val_windows_parsed)
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

    let train_cutoff = cli.train_cutoff.clone();
    let val_windows_parsed = match parse_val_windows(&cli.val_windows) {
        Ok(v) => v,
        Err(e) => { eprintln!("{}", e); return Ok(()); }
    };

    println!("Using train_cutoff={} and {} validation window(s)", train_cutoff, val_windows_parsed.len());
    for (i, (s, e)) in val_windows_parsed.iter().enumerate() {
        println!("  val_{}: {} -> {}", i + 1, s, e);
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
                panic!("Feature column '{}' matches target column '{}' - aborting export", f, t);
            }
        }
    }

    // Skipping full training_data.csv export - this file is not used for model training.
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
    let mut val_rows_per_window: Vec<Vec<&sqlx::postgres::PgRow>> = vec![Vec::new(); val_windows_parsed.len()];
    let mut test_rows = Vec::new();

    // Audit map: ts_code -> [train, val_1, ..., val_N, test, dropped]
    let mut audit_map: HashMap<String, Vec<usize>> = HashMap::new();
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
    // per-validation-window missing counts
    let mut missing_val_per_window: Vec<usize> = vec![0; val_windows_parsed.len()];
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

        // Multi-window splitting by dates:
        // Train: date <= train_cutoff
        // Validation: any of the val_windows_parsed (val_1, val_2, ...)
        // Test: date > last_val_end
        let model_data = &sorted[..];
        // Prepare per-window buckets
        let mut train_part: Vec<&sqlx::postgres::PgRow> = Vec::new();
        let mut val_parts: Vec<Vec<&sqlx::postgres::PgRow>> = vec![Vec::new(); val_windows_parsed.len()];
        let mut test_part: Vec<&sqlx::postgres::PgRow> = Vec::new();
        let mut dropped_in_gaps = 0usize;
        let last_val_end = if !val_windows_parsed.is_empty() { val_windows_parsed.last().unwrap().1.clone() } else { train_cutoff.clone() };

        // per-stock audit counters: train + N vals + test + dropped
        let mut counts: Vec<usize> = vec![0; 1 + val_windows_parsed.len() + 2];
        // indices: 0=train, 1..=N val_i, N+1=test, N+2=dropped

        for row in model_data.iter() {
            let d: String = row.try_get("trade_date").unwrap_or_default();
            if d.as_str() <= train_cutoff.as_str() {
                train_part.push(row);
                counts[0] += 1;
                continue;
            }
            let mut assigned = false;
            for (i, (s, e)) in val_windows_parsed.iter().enumerate() {
                if d.as_str() >= s.as_str() && d.as_str() <= e.as_str() {
                    val_parts[i].push(row);
                    counts[1 + i] += 1;
                    assigned = true;
                    break;
                }
            }
            if assigned { continue; }
            if d.as_str() > last_val_end.as_str() {
                test_part.push(row);
                counts[1 + val_windows_parsed.len()] += 1;
            } else {
                // Gap: between train_cutoff and last_val_end but not in any val window -> drop and count
                dropped_in_gaps += 1;
                counts[1 + val_windows_parsed.len() + 1] += 1;
            }
        }

        // Fallback proportional split if nothing placed into any bin (very sparse history)
        let placed = train_part.len() + test_part.len() + val_parts.iter().map(|v| v.len()).sum::<usize>();
        if placed == 0 {
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
                counts[0] += train_size;
            }
            if val_size > 0 {
                // put into first val window fallback
                val_rows_per_window[0].extend_from_slice(&model_data[train_size..train_size + val_size]);
                counts[1] += val_size;
            }
            if test_size > 0 {
                test_rows.extend_from_slice(&model_data[train_size + val_size..]);
                counts[1 + val_windows_parsed.len()] += test_size;
            }
            partial_coverage_stocks += 1;
            audit_map.insert(ts_code.clone(), counts);
            continue;
        }

        // Track missing bins for diagnostics per-window
        if train_part.is_empty() {
            missing_train += 1;
        }
        for (i, vp) in val_parts.iter().enumerate() {
            if vp.is_empty() {
                missing_val_per_window[i] += 1;
            }
        }
        if test_part.is_empty() {
            missing_test += 1;
        }

        // Use date-split parts
        if !train_part.is_empty() {
            train_rows.extend_from_slice(&train_part);
        }
        for (i, vp) in val_parts.into_iter().enumerate() {
            if !vp.is_empty() {
                val_rows_per_window[i].extend_from_slice(&vp);
            }
        }
        if !test_part.is_empty() {
            test_rows.extend_from_slice(&test_part);
        }

        // Save per-stock audit counts
        audit_map.insert(ts_code.clone(), counts);

        // Track coverage: if a stock has at least required_per_stock days across the window, consider full_coverage
        if placed >= required_per_stock {
            full_coverage_stocks += 1;
        } else {
            partial_coverage_stocks += 1;
        }
    }

    println!("Stocks with full 5y coverage: {}, partial: {}, skipped by --min-years: {}, train_cutoff: {}, validation windows: {}", full_coverage_stocks, partial_coverage_stocks, skipped_by_min_years, train_cutoff, val_windows_parsed.len());
    println!("Stocks missing bins by date-split: missing_train: {}, missing_test: {}, missing_val_per_window: {:?}", missing_train, missing_test, missing_val_per_window);
    if partial_coverage_stocks > 0 {
        println!("Note: {} stocks required proportional fallback splits. Per-stock target days (fallback): {} (train {} val {} test {})", partial_coverage_stocks, required_per_stock, train_days, val_days, test_days);
    }

    // Sanity check: ensure audit per-stock sums for train/val/test match the rows we will write
    let audit_train_total: usize = audit_map.values().map(|c| c.get(0).cloned().unwrap_or(0)).sum();
    let audit_val_totals: Vec<usize> = (0..val_windows_parsed.len()).map(|i| audit_map.values().map(|c| c.get(1 + i).cloned().unwrap_or(0)).sum()).collect();
    let audit_test_total: usize = audit_map.values().map(|c| c.get(1 + val_windows_parsed.len()).cloned().unwrap_or(0)).sum();
    let exported_val_totals: Vec<usize> = val_rows_per_window.iter().map(|v| v.len()).collect();
    if audit_train_total != train_rows.len() || audit_test_total != test_rows.len() || audit_val_totals != exported_val_totals {
        eprintln!("Audit vs export mismatch: train audit {} vs exported {}, val audit {:?} vs exported {:?}, test audit {} vs exported {}", audit_train_total, train_rows.len(), audit_val_totals, exported_val_totals, audit_test_total, test_rows.len());
        return Ok(());
    }

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
    for (i, rows) in val_rows_per_window.iter().enumerate() {
        write_csv(&format!("val_{}.csv", i + 1), rows)?;
    }
    write_csv("test.csv", &test_rows)?;

    // Print counts
    let val_counts: Vec<usize> = val_rows_per_window.iter().map(|v| v.len()).collect();
    println!("Exported train.csv: {} rows", train_rows.len());
    for (i, cnt) in val_counts.iter().enumerate() {
        println!("  val_{}.csv: {} rows", i + 1, cnt);
    }
    println!("Exported test.csv: {} rows", test_rows.len());

    // Write audit CSV: ts_code, train, val_1..val_N, test, dropped, total
    let mut audit_file = File::create("export_audit.csv")?;
    let mut audit_writer = BufWriter::new(&mut audit_file);
    let mut audit_header = vec!["ts_code".to_string(), "train".to_string()];
    for i in 0..val_windows_parsed.len() {
        audit_header.push(format!("val_{}", i + 1));
    }
    audit_header.push("test".to_string());
    audit_header.push("dropped".to_string());
    audit_header.push("total".to_string());
    writeln!(audit_writer, "{}", audit_header.join(","))?;
    for (ts, counts) in &audit_map {
        let mut row = Vec::new();
        row.push(ts.clone());
        // counts length: 1 + N + 2
        let train_cnt = counts.get(0).cloned().unwrap_or(0);
        row.push(train_cnt.to_string());
        for i in 0..val_windows_parsed.len() {
            let v = counts.get(1 + i).cloned().unwrap_or(0);
            row.push(v.to_string());
        }
        let test_cnt = counts.get(1 + val_windows_parsed.len()).cloned().unwrap_or(0);
        let dropped_cnt = counts.get(1 + val_windows_parsed.len() + 1).cloned().unwrap_or(0);
        row.push(test_cnt.to_string());
        row.push(dropped_cnt.to_string());
        let total_cnt: usize = counts.iter().sum();
        row.push(total_cnt.to_string());
        writeln!(audit_writer, "{}", row.join(","))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_default() {
        let v = parse_val_windows(&vec![]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], ("20250301".to_string(), "20251103".to_string()));
    }

    #[test]
    fn parse_two_windows_sorted() {
        let input = vec!["20250701:20251103".to_string(), "20250301:20250630".to_string()];
        let v = parse_val_windows(&input).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, "20250301");
        assert_eq!(v[1].0, "20250701");
    }

    #[test]
    fn parse_invalid_format() {
        let input = vec!["2025030120251103".to_string()];
        assert!(parse_val_windows(&input).is_err());
    }

    #[test]
    fn parse_start_after_end() {
        let input = vec!["20250301:20250228".to_string()];
        assert!(parse_val_windows(&input).is_err());
    }

    #[test]
    fn parse_overlapping() {
        let input = vec!["20250301:20250630".to_string(), "20250615:20251103".to_string()];
        assert!(parse_val_windows(&input).is_err());
    }

    #[test]
    fn parse_contiguous() {
        let input = vec!["20250301:20250630".to_string(), "20250630:20251103".to_string()];
        assert!(parse_val_windows(&input).is_err());
    }
}
