// Export ML training data from ml_training_dataset to CSV, omitting target columns from features.
// Only use Option<f64> safe arithmetic and handle missing data gracefully.

use sqlx::{Column, Row, postgres::PgPoolOptions};
use std::error::Error;
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "export_training_data")]
struct Cli {
    /// Filter stocks listed on or before this date (format: YYYYMMDD). Example: 20200101 to include only stocks listed before 2020-01-01
    #[clap(long)]
    listed_before: Option<String>,

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

    /// Cutoff date for test set in YYYYMMDD format (default: 20251231). Rows after this date are considered test.
    #[clap(long, default_value = "20251231")]
    test_cutoff: String,

    /// Optional ts_code prefix to filter stocks (e.g. "60" for Shanghai A shares)
    #[clap(long)]
    ts_code_prefix: Option<String>,

    /// Optional market type to filter stocks: ZB (60,00), KC (688,30), BJ (92). Overrides ts_code_prefix if specified.
    #[clap(long)]
    market_type: Option<String>,

    /// Optional maximum list_date (stock listing date) in YYYYMMDD format. Only stocks listed on or before this date are included.
    #[clap(long)]
    max_list_date: Option<String>,

    /// Output path for training CSV (default: ./data/training_data.csv)
    #[clap(long, default_value = "./data/training_data.csv")]
    output: String,

    /// Output path for test CSV (default: ./data/test_data.csv)
    #[clap(long, default_value = "./data/test_data.csv")]
    test_output: String,

    /// Train/validation/test split ratios (chronological): space-separated fractions for train, validation, test
    /// Example: "0.6 0.2 0.2" means 60% train, 20% validation, 20% test
    /// Default: "0.7 0.3" means 70% train, 30% validation, 0% test (backward compatible)
    /// Earlier dates go to training, middle dates to validation, later dates to test.
    #[clap(long, default_value = "0.7 0.3")]
    train_val_test_split: String,

    /// Output path for training set (default: ./data/train.csv)
    #[clap(long, default_value = "./data/train.csv")]
    train_output: String,

    /// Output path for validation set (default: ./data/val.csv)
    #[clap(long, default_value = "./data/val.csv")]
    val_output: String,

    /// Output path for test/holdout set (default: ./data/test.csv)
    #[clap(long, default_value = "./data/test.csv")]
    test_holdout_output: String,
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
    if let Some(prefix) = &cli.ts_code_prefix {
        println!("Filtering stocks with ts_code starting with: {}", prefix);
    }

    // Build the SQL selection based on date range
    let min_5day_amount = cli.min_5day_amount;
    let listed_before_cond = if let Some(date) = &cli.listed_before {
        format!("AND s.list_date <= '{}'", date)
    } else {
        "".to_string()
    };
    
    // Handle market_type mapping to ts_code prefixes
    let effective_prefixes = if let Some(market_type) = &cli.market_type {
        match market_type.as_str() {
            "ZB" => vec!["60", "00"],
            "KC" => vec!["688", "30"],
            "BJ" => vec!["92"],
            _ => {
                eprintln!("Invalid market_type: '{}'. Valid values: ZB, KC, BJ", market_type);
                std::process::exit(1);
            }
        }
    } else if let Some(prefix) = &cli.ts_code_prefix {
        vec![prefix.as_str()]
    } else {
        vec![]
    };
    
    // Build ts_code_prefix condition with OR logic for multiple prefixes
    let ts_code_prefix_cond = if !effective_prefixes.is_empty() {
        let prefix_conditions: Vec<String> = effective_prefixes
            .iter()
            .map(|p| format!("s.ts_code LIKE '{}%'", p))
            .collect();
        format!("AND ({})", prefix_conditions.join(" OR "))
    } else {
        "".to_string()
    };
    
    if let Some(market_type) = &cli.market_type {
        println!("Filtering stocks for market type: {} (prefixes: {})", market_type, effective_prefixes.join(", "));
    } else if let Some(prefix) = &cli.ts_code_prefix {
        println!("Filtering stocks with ts_code starting with: {}", prefix);
    }
    
    let max_list_date_cond = if let Some(max_date) = &cli.max_list_date {
        format!("AND s.list_date <= '{}'", max_date)
    } else {
        "".to_string()
    };
    
    if let Some(max_date) = &cli.max_list_date {
        println!("Filtering stocks with list_date on or before: {}", max_date);
    }

    // Quick diagnostics: total rows in ml_training_dataset after test_cutoff and rows for the selected stock set
    let test_cutoff = cli.test_cutoff.clone();

    let total_after_testcutoff: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM ml_training_dataset WHERE trade_date > '{test_cutoff}' AND trade_date <= '{end_date}'",
        test_cutoff = test_cutoff,
        end_date = end_date
    ))
    .fetch_one(&pool)
    .await?;
    println!("Total ml_training_dataset rows with trade_date in ({}, {}]: {}", test_cutoff, end_date, total_after_testcutoff);

    let sel_after_testcutoff: i64 = sqlx::query_scalar(&format!(
        r#"
        WITH eligible AS (
            SELECT s.ts_code
            FROM stock_basic s
            JOIN LATERAL (
                SELECT a.close FROM adjusted_stock_daily a WHERE a.ts_code = s.ts_code ORDER BY a.trade_date DESC LIMIT 1
            ) lc ON lc.close >= 2.0
            WHERE s.list_date <= '{start_date}'
              AND NOT (UPPER(TRIM(s.name)) LIKE 'ST%' OR UPPER(TRIM(s.name)) LIKE '*ST%')
              {listed_before_cond}
              {ts_code_prefix_cond}
              {max_list_date_cond}
              AND EXISTS (
                SELECT 1 FROM ml_training_dataset d WHERE d.ts_code = s.ts_code AND d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
              )
        ),
        recent_liq AS (
            SELECT a.ts_code, AVG(a.amount)::float8 AS avg_amount_3m
            FROM adjusted_stock_daily a
            WHERE a.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '3 months'), 'YYYYMMDD')
            GROUP BY a.ts_code
        ),
        liq_rank AS (
            SELECT r.ts_code, NTILE(100) OVER (ORDER BY COALESCE(r.avg_amount_3m, 0.0) DESC) AS pct_rank
            FROM recent_liq r JOIN eligible e ON e.ts_code = r.ts_code
        ),
        selected_stocks AS (
            SELECT r.ts_code FROM liq_rank r WHERE pct_rank <= 30
        )
        SELECT COUNT(*) FROM ml_training_dataset d JOIN selected_stocks s ON d.ts_code = s.ts_code WHERE d.trade_date > '{test_cutoff}' AND d.trade_date <= '{end_date}'
    "#,
        start_date = start_date,
        end_date = end_date,
        listed_before_cond = listed_before_cond,
        ts_code_prefix_cond = ts_code_prefix_cond,
        max_list_date_cond = max_list_date_cond,
        test_cutoff = test_cutoff
    ))
    .fetch_one(&pool)
    .await?;
    println!("Rows in selected set with trade_date in ({}, {}]: {}", test_cutoff, end_date, sel_after_testcutoff);

    let rows = if min_5day_amount > 0 {
        sqlx::query(&format!(
            r#"
                WITH eligible AS (
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
                      {listed_before_cond}
                      {ts_code_prefix_cond}
                      {max_list_date_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= '{start_date}'
                          AND d.trade_date <= '{end_date}'
                      )
                ),
                recent_liq AS (
                    SELECT a.ts_code,
                           AVG(a.amount)::float8 AS avg_amount_3m,
                           COUNT(*) AS n_days_3m
                    FROM adjusted_stock_daily a
                    WHERE a.trade_date >= TO_CHAR((CURRENT_DATE - INTERVAL '3 months'), 'YYYYMMDD')
                    GROUP BY a.ts_code
                ),
                last_5d AS (
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
                    SELECT r.ts_code FROM liq_rank r JOIN last_5d l ON l.ts_code = r.ts_code WHERE pct_rank <= 30 AND l.amount_5d >= {min_amt}
                )
                SELECT d.*
                FROM ml_training_dataset d
                JOIN selected_stocks s ON d.ts_code = s.ts_code
                WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
                  AND d.next_day_direction IS NOT NULL AND d.next_3day_direction IS NOT NULL
                ORDER BY d.ts_code, d.trade_date
        "#,
            min_amt = min_5day_amount,
            start_date = start_date,
            end_date = end_date,
            listed_before_cond = listed_before_cond,
            ts_code_prefix_cond = ts_code_prefix_cond,
            max_list_date_cond = max_list_date_cond
        ))
        .fetch_all(&pool)
        .await?
    } else {
        sqlx::query(&format!(
            r#"
                WITH eligible AS (
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
                      {listed_before_cond}
                      {ts_code_prefix_cond}
                      {max_list_date_cond}
                      AND EXISTS (
                        SELECT 1 FROM ml_training_dataset d
                        WHERE d.ts_code = s.ts_code
                          AND d.trade_date >= '{start_date}'
                          AND d.trade_date <= '{end_date}'
                      )
                ),
                recent_liq AS (
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
                    SELECT r.ts_code FROM liq_rank r WHERE pct_rank <= 30
                )
                SELECT d.*
                FROM ml_training_dataset d
                JOIN selected_stocks s ON d.ts_code = s.ts_code
                WHERE d.trade_date >= '{start_date}' AND d.trade_date <= '{end_date}'
                  AND d.next_day_direction IS NOT NULL AND d.next_3day_direction IS NOT NULL
                ORDER BY d.ts_code, d.trade_date
        "#, start_date = start_date, end_date = end_date, listed_before_cond = listed_before_cond, ts_code_prefix_cond = ts_code_prefix_cond, max_list_date_cond = max_list_date_cond))
        .fetch_all(&pool)
        .await?
    };

    // Filter out rows where target labels are NULL (next_day_direction or next_3day_direction)
    let mut rows = rows;
    let before_n = rows.len();
    rows.retain(|r| {
        let nd = r.try_get::<Option<i16>, _>("next_day_direction").ok().flatten();
        let n3 = r.try_get::<Option<i16>, _>("next_3day_direction").ok().flatten();
        nd.is_some() && n3.is_some()
    });
    let n_removed_null_targets = before_n - rows.len();
    if n_removed_null_targets > 0 {
        println!("Excluded {} rows with NULL next_day_direction or next_3day_direction", n_removed_null_targets);
    }
    if rows.is_empty() {
        println!("No data found in ml_training_dataset after filtering NULL targets.");
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

    println!("Candidate stocks before filters: {}", total_candidates);
    println!("  Excluded by latest close < 2.0: {}", excluded_by_price);
    println!("  Excluded by name (ST/*ST): {}", excluded_by_name);
    println!("  Excluded by both price & name: {}", excluded_by_both);

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
        .map(|c| c.name().to_string())
        .collect();

    // Inspect for problematic names containing control characters or internal newlines
    for name in &all_colnames {
        if name.contains('\n') || name.contains('\r') {
            eprintln!("Warning: column name contains newline/CR: {:?}", name);
        }
        if name != name.trim() {
            eprintln!("Warning: column name has leading/trailing whitespace: {:?}", name);
        }
    }

    // Target columns (lowercase names used for comparison)
    let target_cols = ["next_day_return", "next_day_direction", "next_3day_return", "next_3day_direction"];

    // Always exclude these from features (metadata)
    let meta_cols: [&str; 2] = ["id", "created_at"];


    // Always include these as identifiers in CSV
    let id_cols = ["ts_code", "trade_date"];

    // Explicitly exclude these features from export
    let removed_features = [
        "kdj_d",
        "macd_weekly_line",
        "industry_momentum",
        "gap_pct",
        "overnight_gap",
        "adx_14",
        "price_roc_10",
        "price_roc_20",
        "bb_middle",
        "price_roc_5",
        "hist_volatility_20",
        "kdj_k",
        "return_lag_1",
        "macd_histogram",
        "macd_monthly_line",
        "macd_line",
        "sma_10",
        "sma_20",
        "bb_upper",
        "quarter",
        "total_share",
        "turnover_rate_f",
        "bb_percent_b",
        "index_csi300_vs_ma20_pct",
        "index_csi300_vs_ma5_pct",
        "atr",
        "index_csi300_pct_chg",
        "volume_accel_5d",
        "bb_lower",
        "quoter",
        "month",
        "sector_momentum_vs_market",
        "fx_usdcnh_pct_chg",
    ];

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
            if removed_features.iter().any(|r| lc == *r) {
                return false;
            }
            true
        })
        .cloned()
        .collect();

    // Final CSV columns: id_cols + feature_cols + categorical embedding cols + target_cols
    let mut csv_cols: Vec<String> = Vec::new();
    for &id in &id_cols {
        csv_cols.push(id.to_string());
    }
    csv_cols.extend_from_slice(&feature_cols);

    // Add embedding column names for industry and act_ent_type
    let mut industry_embed_cols: Vec<String> = Vec::new();
    let mut act_embed_cols: Vec<String> = Vec::new();
    // Fixed embedding dim (8), embeddings are stored in DB columns
    for i in 0..8 {
        let colname = format!("industry_emb_{}", i);
        industry_embed_cols.push(colname.clone());
        csv_cols.push(colname);
    }
    for i in 0..8 {
        let colname = format!("act_ent_type_emb_{}", i);
        act_embed_cols.push(colname.clone());
        csv_cols.push(colname);
    }

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

    // Embeddings are now stored directly in `ml_training_dataset` as columns
    // `industry_emb_0..N` and `act_ent_type_emb_0..N`. We will read those values
    // directly from the row when writing CSV rather than re-hashing here.

    // === CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT ===
    // Create indices and sort by trade_date for chronological split
    let mut row_indices: Vec<usize> = (0..rows.len()).collect();
    row_indices.sort_by(|&i, &j| {
        let date_i: String = rows[i].try_get("trade_date").unwrap_or_default();
        let date_j: String = rows[j].try_get("trade_date").unwrap_or_default();
        date_i.cmp(&date_j)
    });

    // Parse train/val/test split ratios
    let split_ratios: Vec<&str> = cli.train_val_test_split.split_whitespace().collect();
    let (train_ratio, val_ratio, test_ratio) = if split_ratios.len() >= 3 {
        // New format: "train_ratio val_ratio test_ratio"
        let tr: f64 = split_ratios[0].parse().unwrap_or(0.6);
        let vr: f64 = split_ratios[1].parse().unwrap_or(0.2);
        let ter: f64 = split_ratios[2].parse().unwrap_or(0.2);
        (tr, vr, ter)
    } else if split_ratios.len() == 2 {
        // Legacy format: "train_ratio val_ratio" (no test split)
        let tr: f64 = split_ratios[0].parse().unwrap_or(0.7);
        let vr: f64 = split_ratios[1].parse().unwrap_or(0.3);
        (tr, vr, 0.0)
    } else {
        // Single value: backward compatible (treat as train_val_split)
        let tr: f64 = cli.train_val_test_split.parse().unwrap_or(0.7);
        (tr, 1.0 - tr, 0.0)
    };

    // Normalize ratios
    let total = train_ratio + val_ratio + test_ratio;
    let train_ratio = (train_ratio / total).max(0.0).min(1.0);
    let val_ratio = (val_ratio / total).max(0.0).min(1.0);
    let test_ratio = (test_ratio / total).max(0.0).min(1.0);

    // Calculate split by unique trade_date to avoid date overlap between sets
    // Build an ordered list of unique trade_date strings
    let mut unique_dates: Vec<String> = Vec::new();
    let mut last_date: String = String::new();
    for &idx in &row_indices {
        let d: String = rows[idx].try_get("trade_date").unwrap_or_default();
        if d != last_date {
            unique_dates.push(d.clone());
            last_date = d;
        }
    }
    let n_dates = unique_dates.len();
    let train_date_count = ((n_dates as f64) * train_ratio).ceil() as usize;
    let val_date_count = ((n_dates as f64) * val_ratio).ceil() as usize;

    // Determine last dates for each split (inclusive)
    let train_last_date = if train_date_count > 0 { unique_dates[train_date_count - 1].clone() } else { String::new() };
    let val_last_date = if (train_date_count + val_date_count) > 0 && (train_date_count + val_date_count) <= n_dates { unique_dates[(train_date_count + val_date_count) - 1].clone() } else { String::new() };

    // Assign row indices by date boundaries (inclusive for start/end)
    let train_indices: Vec<usize> = row_indices.iter().cloned().filter(|&i| {
        rows[i].try_get::<String, _>("trade_date").unwrap_or_default() <= train_last_date
    }).collect();
    let val_indices: Vec<usize> = row_indices.iter().cloned().filter(|&i| {
        let d = rows[i].try_get::<String, _>("trade_date").unwrap_or_default();
        d > train_last_date && (if val_last_date.len() > 0 { d <= val_last_date } else { true })
    }).collect();
    let test_indices: Vec<usize> = row_indices.iter().cloned().filter(|&i| {
        let d = rows[i].try_get::<String, _>("trade_date").unwrap_or_default();
        if val_last_date.len() > 0 { d > val_last_date } else { d > train_last_date }
    }).collect();

    // Determine date boundaries for reporting
    let train_min_date = if !train_indices.is_empty() {
        rows[train_indices[0]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };
    let train_max_date = if !train_indices.is_empty() {
        rows[train_indices[train_indices.len()-1]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };
    let val_min_date = if !val_indices.is_empty() {
        rows[val_indices[0]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };
    let val_max_date = if !val_indices.is_empty() {
        rows[val_indices[val_indices.len()-1]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };
    let test_min_date = if !test_indices.is_empty() {
        rows[test_indices[0]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };
    let test_max_date = if !test_indices.is_empty() {
        rows[test_indices[test_indices.len()-1]].try_get::<String, _>("trade_date").unwrap_or_default()
    } else {
        "N/A".to_string()
    };

    // Enforce strict chronological non-overlap between splits (fail fast)
    if train_max_date != "N/A" && val_min_date != "N/A" {
        if train_max_date >= val_min_date {
            eprintln!("Error: train.max_date ({}) >= val.min_date ({}). Aborting to avoid date overlap.", train_max_date, val_min_date);
            std::process::exit(1);
        }
    }
    if val_max_date != "N/A" && test_min_date != "N/A" {
        if val_max_date >= test_min_date {
            eprintln!("Error: val.max_date ({}) >= test.min_date ({}). Aborting to avoid date overlap.", val_max_date, test_min_date);
            std::process::exit(1);
        }
    }

    println!("\n📊 CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT");
    println!("Split ratios: {:.1}% Train / {:.1}% Val / {:.1}% Test", 
             train_ratio * 100.0, val_ratio * 100.0, test_ratio * 100.0);
    println!("Training set:   {} rows (dates {} to {})", train_indices.len(), train_min_date, train_max_date);
    println!("Validation set: {} rows (dates {} to {})", val_indices.len(), val_min_date, val_max_date);
    if !test_indices.is_empty() {
        println!("Test/Holdout set: {} rows (dates {} to {})", test_indices.len(), test_min_date, test_max_date);
    }

    if cli.mode == "sliding" {
        // Sliding window mode: export all data without splitting
        println!("Exporting all data to {} (sliding window mode)", cli.output);
        let mut wtr = csv::Writer::from_path(&cli.output)?;
        wtr.write_record(&csv_cols)?;

        for row in &rows {
            let mut record = Vec::new();
            for col in &csv_cols {
                // Embedding columns are stored in DB as `industry_emb_*` and `act_ent_type_emb_*` — read directly
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        println!("Exported {} rows to {}", rows.len(), cli.output);

        // === EXPORT TRAIN/VAL SPLIT (even in sliding mode) ===
        println!("\nExporting chronological train/val split to {} and {}", cli.train_output, cli.val_output);
        
        // Write training set
        let mut train_wtr = csv::Writer::from_path(&cli.train_output)?;
        train_wtr.write_record(&csv_cols)?;
        for &row_idx in &train_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            train_wtr.write_record(&record)?;
        }
        train_wtr.flush()?;
        println!("✅ Exported {} training rows to {}", train_indices.len(), cli.train_output);

        // Write validation set
        let mut val_wtr = csv::Writer::from_path(&cli.val_output)?;
        val_wtr.write_record(&csv_cols)?;
        for &row_idx in &val_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            val_wtr.write_record(&record)?;
        }
        val_wtr.flush()?;
        println!("✅ Exported {} validation rows to {}", val_indices.len(), cli.val_output);

        // Write test/holdout set (if exists)
        if !test_indices.is_empty() {
            let mut test_wtr = csv::Writer::from_path(&cli.test_holdout_output)?;
            test_wtr.write_record(&csv_cols)?;
            for &row_idx in &test_indices {
                let row = &rows[row_idx];
                let mut record = Vec::new();
                for col in &csv_cols {
                    if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                        let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                            Ok(f) => format!("{:.6}", f),
                            Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                                Ok(s) => s,
                                Err(_) => "0.0".to_string(),
                            },
                        };
                        record.push(val);
                        continue;
                    }
                    let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                        v.to_string()
                    } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                        s
                    } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                        b.to_string()
                    } else {
                        "".to_string()
                    };
                    record.push(val);
                }
                test_wtr.write_record(&record)?;
            }
            test_wtr.flush()?;
            println!("✅ Exported {} test/holdout rows to {}", test_indices.len(), cli.test_holdout_output);
        }
    } else {
        // Full mode: export all rows and derive train/val/test splits by chronological split ratios
        println!("Exporting full training data to {} (including all rows)", cli.output);
        let mut wtr = csv::Writer::from_path(&cli.output)?;
        wtr.write_record(&csv_cols)?;

        for row in &rows {
            let mut record = Vec::new();
            for col in &csv_cols {
                // Embedding columns are stored in DB as `industry_emb_*` and `act_ent_type_emb_*` — read directly
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        println!("Exported {} rows to {}", rows.len(), cli.output);

        // === EXPORT TRAIN/VAL/TEST SPLIT ===
        println!("\nExporting chronological train/val/test split to {}, {}, and {}", cli.train_output, cli.val_output, cli.test_holdout_output);

        // Write training set
        let mut train_wtr = csv::Writer::from_path(&cli.train_output)?;
        train_wtr.write_record(&csv_cols)?;
        for &row_idx in &train_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            train_wtr.write_record(&record)?;
        }
        train_wtr.flush()?;
        println!("✅ Exported {} training rows to {}", train_indices.len(), cli.train_output);

        // Write validation set
        let mut val_wtr = csv::Writer::from_path(&cli.val_output)?;
        val_wtr.write_record(&csv_cols)?;
        for &row_idx in &val_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            val_wtr.write_record(&record)?;
        }
        val_wtr.flush()?;
        println!("✅ Exported {} validation rows to {}", val_indices.len(), cli.val_output);

        // Write test/holdout set
        let mut test_wtr = csv::Writer::from_path(&cli.test_holdout_output)?;
        test_wtr.write_record(&csv_cols)?;
        for &row_idx in &test_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            test_wtr.write_record(&record)?;
        }
        test_wtr.flush()?;
        println!("✅ Exported {} test/holdout rows to {}", test_indices.len(), cli.test_holdout_output);
    }

    // Diagnostics for test split
    let test_rows_count = rows.iter().filter(|r| {
        let trade_date: String = r.try_get("trade_date").unwrap_or_default();
        trade_date.as_str() > test_cutoff.as_str()
    }).count();
    let test_rows_in_sel = rows.iter().filter(|r| {
        let trade_date: String = r.try_get("trade_date").unwrap_or_default();
        let ts_code: String = r.try_get("ts_code").unwrap_or_default();
        sel_set.contains(&ts_code) && trade_date.as_str() > test_cutoff.as_str()
    }).count();
    println!("Rows with trade_date > {}: {}", test_cutoff, test_rows_count);
    println!("Rows with trade_date > {} and in selected set: {}", test_cutoff, test_rows_in_sel);

    // Export test data set after test_cutoff into a separate CSV file based on selected stocks
    if cli.mode == "sliding" {
        println!("Exporting test data set to {} (data after {}, strictly based on selected stocks)", cli.test_output, test_cutoff);
        let mut test_wtr = csv::Writer::from_path(&cli.test_output)?;
        test_wtr.write_record(&csv_cols)?;

        for row in &rows {
            let trade_date: String = row.try_get("trade_date").unwrap_or_default();
            let ts_code: String = row.try_get("ts_code").unwrap_or_default();
            if sel_set.contains(&ts_code) && trade_date.as_str() > test_cutoff.as_str() {
                let mut record = Vec::new();
                for col in &csv_cols {
                    // Embedding columns stored in DB — read directly
                    if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                        let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                            Ok(f) => format!("{:.6}", f),
                            Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                                Ok(s) => s,
                                Err(_) => "0.0".to_string(),
                            },
                        };
                        record.push(val);
                        continue;
                    }

                    let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                        v.to_string()
                    } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                        s
                    } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                        i.to_string()
                    } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                        b.to_string()
                    } else {
                        "".to_string()
                    };
                    record.push(val);
                }
                test_wtr.write_record(&record)?;
            }
        }

        test_wtr.flush()?;
        println!("Exported test data set to {}", cli.test_output);
    } else {
        // Full mode: create test file from computed chronological test indices
        println!("Exporting test data set (full mode) to {} from computed test split", cli.test_output);
        let mut test_wtr = csv::Writer::from_path(&cli.test_output)?;
        test_wtr.write_record(&csv_cols)?;
        for &row_idx in &test_indices {
            let row = &rows[row_idx];
            let mut record = Vec::new();
            for col in &csv_cols {
                if col.starts_with("industry_emb_") || col.starts_with("act_ent_type_emb_") {
                    let val: String = match row.try_get::<f64, &str>(col.as_str()) {
                        Ok(f) => format!("{:.6}", f),
                        Err(_) => match row.try_get::<String, &str>(col.as_str()) {
                            Ok(s) => s,
                            Err(_) => "0.0".to_string(),
                        },
                    };
                    record.push(val);
                    continue;
                }
                let val: String = if let Ok(Some(v)) = row.try_get::<Option<f64>, _>(col.as_str()) {
                    v.to_string()
                } else if let Ok(Some(s)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    s
                } else if let Ok(Some(i)) = row.try_get::<Option<i64>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i32>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(i)) = row.try_get::<Option<i16>, _>(col.as_str()) {
                    i.to_string()
                } else if let Ok(Some(b)) = row.try_get::<Option<bool>, _>(col.as_str()) {
                    b.to_string()
                } else {
                    "".to_string()
                };
                record.push(val);
            }
            test_wtr.write_record(&record)?;
        }
        test_wtr.flush()?;
        println!("Exported {} test rows to {}", test_indices.len(), cli.test_output);
    }

    Ok(())
}
