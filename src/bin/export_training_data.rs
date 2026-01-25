// Export ML training data from ml_training_dataset to CSV, omitting target columns from features.
// Only use Option<f64> safe arithmetic and handle missing data gracefully.

use sqlx::{Column, Row, postgres::PgPoolOptions};
use std::error::Error;
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

    /// Start date for data export in YYYYMMDD format (default: 20220701)
    #[clap(long, default_value = "20220701")]
    start_date: String,

    /// End date for data export in YYYYMMDD format (default: current date)
    #[clap(long)]
    end_date: Option<String>,

    /// Export mode: "full" for complete dataset, "sliding" for sliding window compatible (default: full)
    #[clap(long, default_value = "full")]
    mode: String,
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

    // Determine date range
    let start_date = cli.start_date.clone();
    let end_date = cli.end_date.unwrap_or_else(|| {
        // Default to current date
        chrono::Utc::now().format("%Y%m%d").to_string()
    });

    println!("Exporting data from {} to {} in {} mode", start_date, end_date, cli.mode);

    // Build the SQL selection based on date range
    let min_5day_amount = cli.min_5day_amount;
    let min_years_cond = if cli.min_years > 0 {
        // Ensure the stock was listed at least `min_years` before the requested start date
        format!("AND s.list_date <= TO_CHAR((TO_DATE('{}','YYYYMMDD') - INTERVAL '{} years'), 'YYYYMMDD')", start_date, cli.min_years)
    } else {
        "".to_string()
    };

    let rows = if min_5day_amount > 0 {
        sqlx::query(&format!(
            r#"
                WITH eligible AS (
                    -- Stocks with data in the specified date range
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
                    WHERE NOT (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
                      {min_years_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= '{start_date}'
                          AND d.trade_date <= '{end_date}'
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
                    -- Sum traded amount over the last ~5 trading days (approx using 7 calendar days window ending at end_date)
                    SELECT a.ts_code,
                           COALESCE(SUM(a.amount), 0.0)::float8 AS amount_5d
                    FROM adjusted_stock_daily a
                    WHERE a.trade_date >= TO_CHAR((TO_DATE('{end_date}','YYYYMMDD') - INTERVAL '7 days'), 'YYYYMMDD')
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
                WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
                ORDER BY d.ts_code, d.trade_date
        "#,
            min_amt = min_5day_amount,
            start_date = start_date,
            end_date = end_date,
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
                    WHERE s.list_date <= '{start_date}'
                      AND NOT (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
                      {min_years_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= '{start_date}'
                          AND d.trade_date <= '{end_date}'
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
                WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
                ORDER BY d.ts_code, d.trade_date
        "#, start_date = start_date, end_date = end_date, min_years_cond = min_years_cond))
        .fetch_all(&pool)
        .await?
    };


    if rows.is_empty() {
        println!("No data found in ml_training_dataset.");
        return Ok(());
    }

    // Diagnostics: counts of candidate stocks and exclusions by price/name
    // Base candidate universe: stocks listed long enough and with ml_training_dataset rows in last 5 years
    let total_candidates: i64 = sqlx::query_scalar(&format!(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE s.list_date <= '{start_date}'
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= '{start_date}'
              AND d.trade_date <= '{end_date}'
          )
        "#,
        start_date = start_date,
        end_date = end_date
    ))
    .fetch_one(&pool)
    .await?;

    let excluded_by_price: i64 = sqlx::query_scalar(&format!(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        JOIN LATERAL (
            SELECT a.close FROM adjusted_stock_daily a WHERE a.ts_code = s.ts_code ORDER BY a.trade_date DESC LIMIT 1
        ) lc ON lc.close < 2.0
        WHERE s.list_date <= '{start_date}'
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= '{start_date}'
              AND d.trade_date <= '{end_date}'
          )
        "#,
        start_date = start_date,
        end_date = end_date
    ))
    .fetch_one(&pool)
    .await?;

    let excluded_by_name: i64 = sqlx::query_scalar(&format!(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
          AND s.list_date <= TO_CHAR((TO_DATE('{start_date}','YYYYMMDD') - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= '{start_date}'
              AND d.trade_date <= '{end_date}'
          )
        "#,
        start_date = start_date,
        end_date = end_date
    ))
    .fetch_one(&pool)
    .await?;

    let excluded_by_both: i64 = sqlx::query_scalar(&format!(
        r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        JOIN LATERAL (
            SELECT a.close FROM adjusted_stock_daily a WHERE a.ts_code = s.ts_code ORDER BY a.trade_date DESC LIMIT 1
        ) lc ON lc.close < 2.0
        WHERE (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
          AND s.list_date <= TO_CHAR((TO_DATE('{start_date}','YYYYMMDD') - INTERVAL '5 years'), 'YYYYMMDD')
          AND EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date >= '{start_date}'
              AND d.trade_date <= '{end_date}'
          )
        "#,
        start_date = start_date,
        end_date = end_date
    ))
    .fetch_one(&pool)
    .await?;

    let excluded_by_short_history: i64 = if cli.min_years > 0 {
        sqlx::query_scalar(&format!(r#"
        SELECT COUNT(DISTINCT s.ts_code)
        FROM stock_basic s
        WHERE s.list_date <= '{start_date}'
          AND NOT EXISTS (
            SELECT 1 FROM ml_training_dataset d
            WHERE d.ts_code = s.ts_code
              AND d.trade_date <= TO_CHAR((TO_DATE('{start_date}','YYYYMMDD') - INTERVAL '{min_years} years'), 'YYYYMMDD')
          )
        "#, start_date = start_date, min_years = cli.min_years))
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
                panic!("Feature column '{}' matches target column '{}' - aborting export", f, t);
            }
        }
    }

    // Skipping full training_data.csv export - this file is not used for model training.

    // For sliding window mode, export all data without splitting
    println!("Exporting all data to training_data.csv (sliding window mode)");
    let mut wtr = csv::Writer::from_path("training_data.csv")?;
    wtr.write_record(&csv_cols)?;

    for row in &rows {
        let mut record = Vec::new();
        for col in &csv_cols {
            let val: String = match row.try_get::<Option<f64>, _>(col.as_str()) {
                Ok(Some(v)) => v.to_string(),
                Ok(None) => "".to_string(),
                Err(_) => match row.try_get::<Option<String>, _>(col.as_str()) {
                    Ok(Some(s)) => s,
                    Ok(None) => "".to_string(),
                    Err(_) => match row.try_get::<Option<i32>, _>(col.as_str()) {
                        Ok(Some(i)) => i.to_string(),
                        Ok(None) => "".to_string(),
                        Err(_) => "".to_string(),
                    },
                },
            };
            record.push(val);
        }
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    println!("Exported {} rows to training_data.csv", rows.len());

    // Export test data set after 20251231 into a separate CSV file based on selected stocks
    println!("Exporting test data set to test_data.csv (data after 20251231, based on selected stocks)");
    let mut test_wtr = csv::Writer::from_path("test_data.csv")?;
    test_wtr.write_record(&csv_cols)?;

    for row in &rows {
        let trade_date: String = row.try_get("trade_date").unwrap_or_default();
        let ts_code: String = row.try_get("ts_code").unwrap_or_default();
        if trade_date > String::from("20251231") && sel_set.contains(&ts_code) {
            let mut record = Vec::new();
            for col in &csv_cols {
                let val: String = match row.try_get::<Option<f64>, _>(col.as_str()) {
                    Ok(Some(v)) => v.to_string(),
                    Ok(None) => "".to_string(),
                    Err(_) => match row.try_get::<Option<String>, _>(col.as_str()) {
                        Ok(Some(s)) => s,
                        Ok(None) => "".to_string(),
                        Err(_) => match row.try_get::<Option<i32>, _>(col.as_str()) {
                            Ok(Some(i)) => i.to_string(),
                            Ok(None) => "".to_string(),
                            Err(_) => "".to_string(),
                        },
                    },
                };
                record.push(val);
            }
            test_wtr.write_record(&record)?;
        }
    }

    test_wtr.flush()?;
    println!("Exported test data set to test_data.csv");

    Ok(())
}
