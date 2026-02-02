use chrono::{Datelike, NaiveDate};
use rust_llm_stock::{
    bollinger::BollingerBands, kdj::KDJIndicator, stock_db::get_connection, ts::model::DailyModel,
};
use sqlx::{Pool, Postgres};
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use clap::Parser;
use futures::stream::{self, StreamExt};
use ta::{
    DataItem, Next,
    indicators::{
        AverageTrueRange as Atr, RelativeStrengthIndex as Rsi, SimpleMovingAverage as Sma,
    },
};

/// Embedding defaults (fixed constants)
const EMBED_DIM: usize = 8;
const EMBED_SEED: u64 = 42;


/// Struct to hold daily data (raw values)
#[derive(Debug, Clone)]
struct AdjustedDailyData {
    ts_code: String,
    trade_date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    amount: Option<f64>,
    pct_chg: Option<f64>,
}

/// Fetch adjusted daily data for a stock (sync, blocking)
/// Fetch adjusted daily data for a stock (async, takes pool argument)
async fn fetch_adjusted_daily_data(
    pool: &sqlx::Pool<sqlx::Postgres>,
    ts_code: &str,
    min_date: &str,
    max_date: &str,
) -> Vec<AdjustedDailyData> {
    sqlx::query_as!(
        AdjustedDailyData,
        "SELECT COALESCE(ts_code, '') as \"ts_code!\",
            COALESCE(trade_date, '') as \"trade_date!\",
            COALESCE(open::DOUBLE PRECISION, 0.0) as \"open!\",
            COALESCE(high::DOUBLE PRECISION, 0.0) as \"high!\",
            COALESCE(low::DOUBLE PRECISION, 0.0) as \"low!\",
            COALESCE(close::DOUBLE PRECISION, 0.0) as \"close!\",
            COALESCE(volume::DOUBLE PRECISION, 0.0) as \"volume!\",
            amount::DOUBLE PRECISION,
            pct_chg::DOUBLE PRECISION
         FROM adjusted_stock_daily WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3 ORDER BY trade_date ASC",
        ts_code,
        min_date,
        max_date
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default()
}

/// Calculate EMA for custom periods
fn calculate_ema_custom(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return prices.last().copied().unwrap_or(0.0);
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let sma: f64 = prices[prices.len() - period..].iter().sum::<f64>() / period as f64;

    let mut ema = sma;
    for &price in &prices[prices.len() - period..] {
        ema = (price - ema) * multiplier + ema;
    }

    ema
}

/// Calculate MACD for custom timeframes (weekly, monthly)
fn calculate_macd_custom(prices: &[f64], fast: usize, slow: usize) -> (Option<f64>, Option<f64>) {
    if prices.len() < slow {
        return (None, None);
    }

    let fast_ema = calculate_ema_custom(prices, fast);
    let slow_ema = calculate_ema_custom(prices, slow);
    let macd_line = fast_ema - slow_ema;
    let signal_line = macd_line * 0.85; // Simplified signal approximation

    (Some(macd_line), Some(signal_line))
}

/// Calculate ASI (Accumulation Swing Index) for adjusted daily data
fn calculate_asi_adjusted(quotes: &[&AdjustedDailyData], limit_move: f64) -> f64 {
    if quotes.len() < 2 {
        return 0.0;
    }

    let mut asi_sum = 0.0;

    for i in 1..quotes.len() {
        let curr = quotes[i];
        let prev = quotes[i - 1];

        let close = curr.close;
        let open = curr.open;
        let high = curr.high;
        let low = curr.low;
        let prev_close = prev.close;
        let prev_open = prev.open;

        let a = (high - prev_close).abs();
        let b = (low - prev_close).abs();
        let c = (high - low).abs();
        let d = (prev_close - prev_open).abs();

        let k = a.max(b);

        let r = if a >= b && a >= c {
            a - 0.5 * b + 0.25 * d
        } else if b >= a && b >= c {
            b - 0.5 * a + 0.25 * d
        } else {
            c + 0.25 * d
        };

        let si = if r != 0.0 && limit_move != 0.0 {
            50.0 * ((close - prev_close) + 0.5 * (close - open) + 0.25 * (prev_close - prev_open))
                / r
                * (k / limit_move)
        } else {
            0.0
        };

        asi_sum += si;
    }

    asi_sum
}

/// Calculate OBV (On Balance Volume) for adjusted daily data
fn calculate_obv_adjusted(quotes: &[&AdjustedDailyData]) -> f64 {
    if quotes.len() < 2 {
        return 0.0;
    }

    let mut obv = 0.0;

    for i in 1..quotes.len() {
        let curr_close = quotes[i].close;
        let prev_close = quotes[i - 1].close;
        let volume = quotes[i].volume;

        if curr_close > prev_close {
            obv += volume;
        } else if curr_close < prev_close {
            obv -= volume;
        }
    }

    obv
}

/// Calculate ASI (Accumulation Swing Index)
fn calculate_asi(quotes: &[&DailyModel], limit_move: f64) -> f64 {
    if quotes.len() < 2 {
        return 0.0;
    }

    let mut asi_sum = 0.0;

    for i in 1..quotes.len() {
        let curr = quotes[i];
        let prev = quotes[i - 1];

        if let (Some(close), Some(open), Some(high), Some(low), Some(prev_close), Some(prev_open)) = (
            curr.close, curr.open, curr.high, curr.low, prev.close, prev.open,
        ) {
            let a = (high - prev_close).abs();
            let b = (low - prev_close).abs();
            let c = (high - low).abs();
            let d = (prev_close - prev_open).abs();

            let k = a.max(b);

            let r = if a >= b && a >= c {
                a - 0.5 * b + 0.25 * d
            } else if b >= a && b >= c {
                b - 0.5 * a + 0.25 * d
            } else {
                c + 0.25 * d
            };

            let si = if r != 0.0 && limit_move != 0.0 {
                50.0 * ((close - prev_close)
                    + 0.5 * (close - open)
                    + 0.25 * (prev_close - prev_open))
                    / r
                    * (k / limit_move)
            } else {
                0.0
            };

            asi_sum += si;
        }
    }

    asi_sum
}

/// Calculate OBV (On Balance Volume)
fn calculate_obv(quotes: &[&DailyModel]) -> f64 {
    if quotes.len() < 2 {
        return 0.0;
    }

    let mut obv = 0.0;

    for i in 1..quotes.len() {
        if let (Some(curr_close), Some(prev_close), Some(volume)) =
            (quotes[i].close, quotes[i - 1].close, quotes[i].vol)
        {
            if curr_close > prev_close {
                obv += volume;
            } else if curr_close < prev_close {
                obv -= volume;
            }
        }
    }

    obv
}

/// Extract time features from trade date
fn extract_time_features(trade_date: &str) -> (Option<i16>, Option<i16>, Option<i16>, Option<i16>) {
    if let Ok(date) = NaiveDate::parse_from_str(trade_date, "%Y%m%d") {
        let month = date.month() as i16;
        let weekday = date.weekday().num_days_from_monday() as i16;
        let quarter = ((month - 1) / 3 + 1) as i16;
        let week_no = date.iso_week().week() as i16;
        (Some(month), Some(weekday), Some(quarter), Some(week_no))
    } else {
        (None, None, None, None)
    }
}

/// Round a number to 2 decimal places
fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

/// Calculate ADX (Average Directional Index)
fn calculate_adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return 0.0;
    }

    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;

    for i in 1..=period {
        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];

        let plus_dm = if high_diff > low_diff && high_diff > 0.0 {
            high_diff
        } else {
            0.0
        };
        let minus_dm = if low_diff > high_diff && low_diff > 0.0 {
            low_diff
        } else {
            0.0
        };

        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());

        plus_dm_sum += plus_dm;
        minus_dm_sum += minus_dm;
        tr_sum += tr;
    }

    if tr_sum == 0.0 {
        return 0.0;
    }

    let plus_di = 100.0 * plus_dm_sum / tr_sum;
    let minus_di = 100.0 * minus_dm_sum / tr_sum;

    if plus_di + minus_di == 0.0 {
        return 0.0;
    }

    100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
}

/// Calculate VWAP (Volume Weighted Average Price) approximation for the day
fn calculate_vwap(
    highs: &[Option<f64>],
    lows: &[Option<f64>],
    closes: &[Option<f64>],
    volumes: &[Option<f64>],
    lookback: usize,
) -> f64 {
    let len = highs.len();
    if len == 0 {
        return 0.0;
    }

    let start = if len > lookback { len - lookback } else { 0 };
    let mut cum_pv = 0.0;
    let mut cum_vol = 0.0;

    for i in start..len {
        let typical_price =
            (highs[i].unwrap_or(0.0) + lows[i].unwrap_or(0.0) + closes[i].unwrap_or(0.0)) / 3.0;
        let volume = volumes[i].unwrap_or(0.0);
        cum_pv += typical_price * volume;
        cum_vol += volume;
    }

    if cum_vol > 0.0 {
        cum_pv / cum_vol
    } else {
        closes.last().and_then(|v| v.clone()).unwrap_or(0.0)
    }
}

/// Calculate Chaikin Money Flow (CMF)
fn calculate_cmf(
    highs: &[Option<f64>],
    lows: &[Option<f64>],
    closes: &[Option<f64>],
    volumes: &[Option<f64>],
    period: usize,
) -> f64 {
    if highs.len() < period {
        return 0.0;
    }

    let len = highs.len();
    let start = len - period;

    let mut mfv_sum = 0.0;
    let mut vol_sum = 0.0;

    for i in start..len {
        let high = highs[i].unwrap_or(0.0);
        let low = lows[i].unwrap_or(0.0);
        let close = closes[i].unwrap_or(0.0);
        let volume = volumes[i].unwrap_or(0.0);
        let hl_range = high - low;
        if hl_range > 0.0 {
            let mf_multiplier = ((close - low) - (high - close)) / hl_range;
            let mfv = mf_multiplier * volume;
            mfv_sum += mfv;
        }
        vol_sum += volume;
    }

    if vol_sum > 0.0 {
        mfv_sum / vol_sum
    } else {
        0.0
    }
}

/// Calculate Money Flow Index (MFI)
fn calculate_mfi(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    period: usize,
) -> f64 {
    // Need period + 1 data points to compare previous typical price
    if closes.len() < period + 1 || highs.len() < period + 1 || lows.len() < period + 1 || volumes.len() < period + 1 {
        return 50.0; // neutral default
    }

    let len = closes.len();
    let start = len - period; // include this..len-1

    let mut positive_mf = 0.0;
    let mut negative_mf = 0.0;

    for i in start..len {
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        let prev_tp = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3.0;
        let mf = tp * volumes[i];
        if tp > prev_tp {
            positive_mf += mf;
        } else if tp < prev_tp {
            negative_mf += mf;
        }
    }

    if negative_mf.abs() < 1e-12 {
        return 100.0;
    }

    let mfr = positive_mf / negative_mf;
    100.0 - (100.0 / (1.0 + mfr))
}

/// Calculate Williams %R
fn calculate_williams_r(highs: &[f64], lows: &[f64], close: f64, period: usize) -> f64 {
    if highs.len() < period {
        return -50.0;
    }

    let len = highs.len();
    let start = len - period;

    let highest = highs[start..]
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = lows[start..].iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if highest > lowest {
        -100.0 * (highest - close) / (highest - lowest)
    } else {
        -50.0
    }
}

/// Calculate Aroon Indicator (returns Aroon Up, Aroon Down)
fn calculate_aroon(highs: &[f64], lows: &[f64], period: usize) -> (f64, f64) {
    if highs.len() < period {
        return (50.0, 50.0);
    }

    let len = highs.len();
    let start = len - period;

    let mut high_idx = start;
    let mut low_idx = start;

    for i in start..len {
        if highs[i] >= highs[high_idx] {
            high_idx = i;
        }
        if lows[i] <= lows[low_idx] {
            low_idx = i;
        }
    }

    let periods_since_high = (len - 1 - high_idx) as f64;
    let periods_since_low = (len - 1 - low_idx) as f64;

    let aroon_up = 100.0 * (period as f64 - periods_since_high) / period as f64;
    let aroon_down = 100.0 * (period as f64 - periods_since_low) / period as f64;

    (aroon_up, aroon_down)
}

/// Detect Doji candlestick pattern
fn is_doji(open: Option<f64>, close: Option<f64>, high: Option<f64>, low: Option<f64>) -> bool {
    let open = open.unwrap_or(0.0);
    let close = close.unwrap_or(0.0);
    let high = high.unwrap_or(0.0);
    let low = low.unwrap_or(0.0);
    let body = (close - open).abs();
    let range = high - low;
    if range > 0.0 {
        body / range < 0.1 // Body is less than 10% of range
    } else {
        false
    }
}

/// Detect Hammer candlestick pattern
fn is_hammer(open: Option<f64>, close: Option<f64>, high: Option<f64>, low: Option<f64>) -> bool {
    let open = open.unwrap_or(0.0);
    let close = close.unwrap_or(0.0);
    let high = high.unwrap_or(0.0);
    let low = low.unwrap_or(0.0);
    let body = (close - open).abs();
    let lower_wick = open.min(close) - low;
    let upper_wick = high - open.max(close);

    if body > 0.0 {
        lower_wick > 2.0 * body && upper_wick < body
    } else {
        false
    }
}

/// Detect Shooting Star candlestick pattern
fn is_shooting_star(
    open: Option<f64>,
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
) -> bool {
    let open = open.unwrap_or(0.0);
    let close = close.unwrap_or(0.0);
    let high = high.unwrap_or(0.0);
    let low = low.unwrap_or(0.0);
    let body = (close - open).abs();
    let upper_wick = high - open.max(close);
    let lower_wick = open.min(close) - low;

    if body > 0.0 {
        upper_wick > 2.0 * body && lower_wick < body
    } else {
        false
    }
}

/// Count consecutive up/down days
fn count_consecutive_days(closes: &[f64]) -> i32 {
    if closes.len() < 2 {
        return 0;
    }

    let len = closes.len();
    let mut count = 0;
    let is_up = closes[len - 1] > closes[len - 2];

    for i in (1..len).rev() {
        if is_up && closes[i] > closes[i - 1] {
            count += 1;
        } else if !is_up && closes[i] < closes[i - 1] {
            count -= 1;
        } else {
            break;
        }
    }

    count
}

#[derive(Parser, Debug)]
#[command(name = "dataset_creator")]
struct Cli {
    /// Start date (YYYYMMDD) for incremental or test runs
    #[arg(long)]
    start_date: Option<String>,

    /// End date (YYYYMMDD) for incremental or test runs
    #[arg(long)]
    end_date: Option<String>,

    /// Concurrency level (number of stocks processed concurrently)
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Dry run: don't insert into DB (for testing)
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Verbose: print debug diagnostic messages (disabled by default)
    #[arg(long, default_value_t = false)]
    verbose: bool,
    /// Leak check mode: run a single-stock leakage validation and abort on leak
    #[arg(long, default_value_t = false)]
    leak_check: bool,

    /// Stock code to run leak-check on (optional). If not provided, first stock is used.
    #[arg(long)]
    leak_stock: Option<String>,

    /// Number of years of history to include per stock (default: 7). Use at least 7 to cover a 6.5-year train/val/test split.
    #[arg(long, default_value_t = 7)]
    history_years: usize,

    /// Apply monthly cross-sectional winsorization to `next_day_return` after dataset creation
    #[arg(long, default_value_t = false)]
    winsorize_monthly: bool,

    /// Winsorization tail percentile (e.g., 0.025 for 2.5%)
    #[arg(long, default_value_t = 0.025)]
    winsorize_pct: f64,

    /// Winsorize selected heavy-tailed features before insertion (requires concurrency=1)
    #[arg(long, default_value_t = false)]
    winsorize_features: bool,

    /// Compute VIF across numeric features and DROP columns with VIF > 10 (memory heavy; collects full dataset). Only supported with concurrency=1.
    #[arg(long, default_value_t = false)]
    drop_high_vif: bool,

    /// Apply log1p transform to selected skewed positive features before scaling (requires concurrency=1 if used)
    #[arg(long, default_value_t = false)]
    apply_log1p: bool,

    /// Standardize (z-score) numeric features across the dataset before insertion (requires concurrency=1 if used)
    #[arg(long, default_value_t = false)]
    standardize: bool,
} 

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();
    eprintln!("[DEBUG] CLI args: start_date={:?}, end_date={:?}, concurrency={}, dry_run={}", cli.start_date, cli.end_date, cli.concurrency, cli.dry_run);
    let start_time = std::time::Instant::now();
    let dbpool = get_connection().await;

    // Create the training dataset table if not exists
    create_ml_training_dataset_table(&dbpool).await?;

    // Get ALL stocks from stock_basic table
    // Process full date range available in adjusted_stock_daily table
    let mut stocks = sqlx::query_as::<_, (String, Option<String>, Option<String>, Option<String>)>(
        r#"
        SELECT DISTINCT 
            sb.ts_code,
            sb.list_date,
            sb.industry,
            sb.act_ent_type
        FROM stock_basic sb
        WHERE EXISTS (
            SELECT 1 FROM adjusted_stock_daily d 
            WHERE d.ts_code = sb.ts_code
        )
        AND sb.ts_code IS NOT NULL
        AND sb.name NOT LIKE 'ST%'
        AND sb.name NOT LIKE '%ST'
        ORDER BY sb.ts_code
        "#,
    )
    .fetch_all(&dbpool)
    .await?;

    println!("Processing {} stocks for dataset creation", stocks.len());
    eprintln!("[DEBUG] Step 1: After printing stock count");

    // Debug: print a small sample of stocks to verify act_ent_type is fetched from stock_basic
    for (i, (ts_code, list_date, industry, act_ent_type)) in stocks.iter().take(10).enumerate() {
        eprintln!(
            "[DEBUG] Sample stock {}: ts_code={}, list_date={:?}, industry={:?}, act_ent_type={:?}",
            i, ts_code, list_date, industry, act_ent_type
        );
    }

    let processing_start = std::time::Instant::now();

    // If leak_check requested, force sequential processing. Only restrict to a single
    // stock if `--leak-stock` is provided; otherwise run leak-check across all stocks.
    let concurrency = if cli.leak_check { 1usize } else { cli.concurrency };
    if cli.leak_check {
        if let Some(ts) = cli.leak_stock.clone() {
            stocks.retain(|(code, _, _, _)| code == &ts);
            if stocks.is_empty() {
                eprintln!("⚠️  leak-check: specified stock {} not found in stock list", ts);
                std::process::exit(3);
            }
            println!("Running leak-check for {} (concurrency forced to 1)", ts);
        } else {
            println!("Running leak-check for all stocks (concurrency forced to 1)");
        }
    }

    // Check max date in ml_training_dataset for incremental backfill
    eprintln!("[DEBUG] Step 2: About to query max_ml_date");
    let max_ml_date: Option<String> =
        sqlx::query_scalar("SELECT COALESCE(MAX(trade_date), '') FROM ml_training_dataset")
            .fetch_optional(&dbpool)
            .await?
            .flatten()
            .filter(|s: &String| !s.is_empty());
    eprintln!("[DEBUG] Step 3: Got max_ml_date = {:?}", max_ml_date);

    // For incremental updates, start from the day AFTER max_ml_date, or use CLI --start-date if provided
    let incremental_start: Option<String> = if let Some(ref s) = cli.start_date {
        println!("⚠️  CLI start-date override provided: {}", s);
        Some(s.clone())
    } else if let Some(ref ml_date) = max_ml_date {
        let date = chrono::NaiveDate::parse_from_str(ml_date, "%Y%m%d")?;
        let next_date = date + chrono::Duration::days(1);
        Some(next_date.format("%Y%m%d").to_string())
    } else {
        None
    };

    // Get the date range from stock_daily table (more comprehensive than index_daily)
    // This handles trading holidays better - stock_daily has all actual trading dates
    let date_range = sqlx::query_as::<_, (String, String)>(
        "SELECT 
    COALESCE(GREATEST(
        TO_CHAR(TO_DATE(MIN(trade_date), 'YYYYMMDD') - INTERVAL '750 days', 'YYYYMMDD'), 
        COALESCE($1, '20110101')
    ), '20110101') AS min_date, 
    COALESCE(MAX(trade_date), '20251231') AS max_date 
 FROM stock_daily",
    )
    .bind(&incremental_start) // Use day after max_ml_date for incremental updates
    .fetch_one(&dbpool)
    .await?;

    let min_date = date_range.0;
    let max_date = date_range.1;

    // Fix: If incremental start date is after max_date, do a full rebuild instead
    let (mut final_min_date, final_max_date) = if min_date > max_date {
        println!(
            "⚠️  Incremental update would create invalid range ({} to {}), doing full rebuild instead",
            min_date, max_date
        );
        // Calculate a reasonable start date for full rebuild (750 days back from max)
        let full_start = chrono::NaiveDate::parse_from_str(&max_date, "%Y%m%d")
            .ok()
            .and_then(|d| {
                Some(
                    (d - chrono::Duration::days(750))
                        .format("%Y%m%d")
                        .to_string(),
                )
            })
            .unwrap_or_else(|| "20110101".to_string());
        (full_start, max_date.clone())
    } else {
        (min_date.clone(), max_date.clone())
    };

    if let Some(ref ml_date) = max_ml_date {
        println!(
            "Incremental update: existing data up to {}, processing {} to {}",
            ml_date, final_min_date, final_max_date
        );
    } else {
        println!(
            "Full dataset creation: {} to {} (constrained to global indices availability)",
            final_min_date, final_max_date
        );
    }

    // Enforce history window per stock based on CLI `history_years` (default 5 years)
    // This moves the start date forward so we only create features within the recent window.
    let history_years = cli.history_years;
    if history_years > 0 {
        if let Ok(max_dt) = chrono::NaiveDate::parse_from_str(&final_max_date, "%Y%m%d") {
            let hist_days = (365 * history_years) as i64;
            let desired_start = (max_dt - chrono::Duration::days(hist_days)).format("%Y%m%d").to_string();
            if desired_start.as_str() > final_min_date.as_str() {
                println!("⚠️  Limiting history to last {} years: {} -> {}", history_years, desired_start, final_max_date);
                final_min_date = desired_start;
            }
        }
    }

    // Warn if configured history is less than recommended for export (6.5 years required).
    if history_years < 7 {
        println!("⚠️  Configured --history-years={} is less than 7 years. To ensure full coverage for a 5y train + 1y val + 0.5y test split, run with --history-years 7 or higher.", history_years);
    }

    // Allow CLI overrides for testing: if --start-date or --end-date provided, force the range
    let mut final_min_date = final_min_date;
    let mut final_max_date = final_max_date;
    if let Some(ref s) = cli.start_date {
        final_min_date = s.clone();
        println!("⚠️  CLI override: forcing start_date = {}", final_min_date);
    }
    if let Some(ref e) = cli.end_date {
        final_max_date = e.clone();
        println!("⚠️  CLI override: forcing end_date = {}", final_max_date);
    }

    // Compute effective current date (today) and clamp to data availability window
    let system_date = chrono::Local::now().format("%Y%m%d").to_string();
    if final_max_date.as_str() > system_date.as_str() {
        println!("⚠️  final_max_date {} exceeds system date {}. Clamping to system date.", final_max_date, system_date);
        final_max_date = system_date.clone();
    }

    // Ensure max_date (from DB) does not exceed system date
    let max_date = if max_date.as_str() > system_date.as_str() {
        println!("⚠️  max_date {} exceeds system date {}. Clamping to system date.", max_date, system_date);
        system_date.clone()
    } else {
        max_date.clone()
    };

    // fetch_start should also not be beyond system_date
    let fetch_start = chrono::NaiveDate::parse_from_str(&min_date, "%Y%m%d")
        .ok()
        .and_then(|d| {
            let adjusted_date = (d - chrono::Duration::days(400))
                .format("%Y%m%d")
                .to_string();
            if adjusted_date.as_str() > system_date.as_str() {
                println!("⚠️  fetch_start {} exceeds system date {}. Adjusting to {}.", adjusted_date, system_date, system_date);
                Some(system_date.clone())
            } else {
                Some(adjusted_date)
            }
        })
        .unwrap_or_else(|| "20100101".to_string());

    // effective_current_date: the latest date we consider "current" for leakage checks
    let effective_current_date = final_max_date.clone();

    // Pre-fetch industry performance data once for all stocks (for full date range)
    // Note: We need data from dates BEFORE final_min_date for prior-day lookups,
    // so extend the prefetch range backward by 10 trading days (~14 calendar days)
    let industry_prefetch_start = chrono::NaiveDate::parse_from_str(&final_min_date, "%Y%m%d")
        .ok()
        .map(|d| (d - chrono::Duration::days(14)).format("%Y%m%d").to_string())
        .unwrap_or_else(|| final_min_date.clone());
    println!("\n=== Pre-fetching Industry Performance Data ({} to {}) ===", industry_prefetch_start, final_max_date);
    let industry_perf_data =
        std::sync::Arc::new(prefetch_industry_performance(&dbpool, &industry_prefetch_start, &final_max_date).await);


    // Prefetch index data for all required indices
    let index_codes = [
        "000300.SH",   // CSI300
        "399006.SZ",   // ChiNext
        "XIN9",        // XIN9
        "HSI",         // Hong Kong Hang Seng Index
        "USDCNH.FXCM", // USD/CNH exchange rate
    ];
    let index_data =
        prefetch_index_data(&dbpool, &index_codes, &final_min_date, &final_max_date).await;

    // Process stocks with optional concurrency
    let total_stocks = stocks.len();

    // If requested, collect all rows for VIF computation before inserting
    let mut all_rows: Vec<FeatureRow> = Vec::new();
    if (cli.drop_high_vif || cli.apply_log1p || cli.standardize || cli.winsorize_features) && cli.concurrency != 1 {
        eprintln!("Error: --drop-high-vif/--apply-log1p/--standardize/--winsorize-features require concurrency=1 (to collect rows deterministically)");
        std::process::exit(2);
    } 

    // Final counters (populated by selected execution path)
    let mut final_processed: usize = 0;
    let mut final_skipped: usize = 0;
    let mut final_records: usize = 0;

    if concurrency <= 1 {
        println!("\n=== Processing {} stocks (sequential) ===\n", total_stocks);

        // Simple counters for progress tracking
        let mut processed_count = 0;
        let mut skipped_count = 0;
        let mut total_records = 0;

        // Process each stock one at a time
        for (idx, (ts_code, list_date, industry, act_ent_type)) in stocks.iter().enumerate() {
            let stock_timer = std::time::Instant::now();

            let calc_start = std::time::Instant::now();
            let feature_rows = calculate_features_for_stock_sync(
                &dbpool,
                ts_code,
                list_date.as_deref(),
                industry.as_deref(),
                act_ent_type.as_deref(),
                &industry_perf_data,
                &final_min_date,
                &final_max_date,
                &index_data,
                history_years,
                cli.verbose,
                cli.leak_check,
                &effective_current_date,
            )
            .await; 
            let calc_elapsed = calc_start.elapsed().as_millis();

            if feature_rows.is_empty() {
                skipped_count += 1;
                eprintln!("  ⚠️  Stock {} skipped (calc: {}ms)", ts_code, calc_elapsed);
            } else {
                // Insert data for this stock
                let row_count = feature_rows.len();

                let insert_start = std::time::Instant::now();
                if !cli.dry_run {
                    if cli.drop_high_vif || cli.apply_log1p || cli.standardize || cli.winsorize_features {
                        // Collect rows for a single global transform/VIF pass
                        all_rows.extend(feature_rows);
                    } else {
                        batch_insert_feature_rows(&dbpool, &feature_rows).await?;
                    }
                }
                let insert_elapsed = insert_start.elapsed().as_millis();

                processed_count += 1;
                total_records += row_count;

                let total_elapsed = stock_timer.elapsed().as_millis();

                // Progress logging every 10 stocks for visibility
                if processed_count % 10 == 0 || processed_count <= 100 {
                    println!(
                        "  [{}/{}] {} processed: {} rows in {}ms (calc: {}ms, insert: {}ms) - {} total committed",
                        processed_count,
                        total_stocks,
                        ts_code,
                        row_count,
                        total_elapsed,
                        calc_elapsed,
                        insert_elapsed,
                        total_records
                    );
                } else if processed_count % 5 == 0 {
                    // Lightweight logging every 5 stocks
                    println!(
                        "  [{}/{}] {} - {}ms",
                        processed_count, total_stocks, ts_code, total_elapsed
                    );
                }
            }
        }

        final_processed = processed_count;
        final_skipped = skipped_count;
        final_records = total_records;

        println!(
            "\n✅ Processed {} stocks, inserted {} records in {:?}",
            final_processed,
            final_records,
            start_time.elapsed()
        );
        if final_skipped > 0 {
            println!(
                "⚠️  Skipped {} stocks due to insufficient historical data (< 60 days)",
                final_skipped
            );
        }

        // If requested, optionally apply log1p and/or standardization to collected rows, then compute VIF and/or insert
        if cli.drop_high_vif || cli.apply_log1p || cli.standardize || cli.winsorize_features {
            println!("⚙️  Preparing collected rows (n_rows={})...", all_rows.len());

            // Apply winsorization (always within apply_log1p_and_standardize), and optionally log1p/standardize
            println!("🔧 Applying winsorization and optional log1p/standard scaling...");
            // Always run transformations so artifacts (scale params) are produced during dry-run for inspection
            apply_log1p_and_standardize(&mut all_rows, cli.apply_log1p, cli.standardize)?;
            println!("✅ Transformations applied. scale params saved to artifacts/scale_params.csv (if standardize enabled)");
            if cli.dry_run {
                println!("Dry-run: insertion will be skipped, but transforms and artifacts are written.");
            }

            if cli.drop_high_vif {
                println!("⚙️  Computing VIF across collected rows (n_rows={})...", all_rows.len());
                // Generate VIF report and SQL; when dry-run, do not execute ALTER statements
                compute_vif_and_drop_columns(&dbpool, &all_rows, !cli.dry_run).await?;
                if cli.dry_run {
                    println!("Dry-run: VIF report and SQL generated, no schema changes executed.");
                }
            }

            // Insert all rows in a batch (if not dry-run & no VIF-only dry-run)
            if !cli.dry_run {
                batch_insert_feature_rows(&dbpool, &all_rows).await?;
                final_records = all_rows.len();
                println!("✅ Inserted {} rows after transforms.", final_records);
            }
        }
    } else {
        println!("\n=== Processing {} stocks (concurrency={}) ===\n", total_stocks, concurrency);

        let processed = Arc::new(AtomicUsize::new(0));
        let skipped = Arc::new(AtomicUsize::new(0));
        let total_records_atomic = Arc::new(AtomicUsize::new(0));

        let stock_stream = stream::iter(stocks.into_iter())
            .map(|(ts_code, list_date, industry, act_ent_type)| {
                let dbpool = dbpool.clone();
                let industry_perf_data = industry_perf_data.clone();
                let index_data = index_data.clone();
                let final_min_date = final_min_date.clone();
                let final_max_date = final_max_date.clone();
                let processed = processed.clone();
                let skipped = skipped.clone();
                let total_records_atomic = total_records_atomic.clone();
                let dry_run = cli.dry_run;
                let verbose = cli.verbose;
                // Clone the effective_current_date for use inside the async move block
                let effective_current_date_value = effective_current_date.clone();

                async move {
                    let calc_start = std::time::Instant::now();
                    let feature_rows = calculate_features_for_stock_sync(
                        &dbpool,
                        &ts_code,
                        list_date.as_deref(),
                        industry.as_deref(),
                        act_ent_type.as_deref(),
                        &industry_perf_data,
                        &final_min_date,
                        &final_max_date,
                        &index_data,
                        history_years,
                        verbose,
                        cli.leak_check,
                        &effective_current_date_value,
                    )
                    .await; 
                    let calc_elapsed = calc_start.elapsed().as_millis();

                    if feature_rows.is_empty() {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        eprintln!("  ⚠️  Stock {} skipped (calc: {}ms)", ts_code, calc_elapsed);
                        return Ok::<(), Box<dyn Error + Send + Sync>>(());
                    }

                    if !dry_run {
                        batch_insert_feature_rows(&dbpool, &feature_rows).await?;
                    }

                    let row_count = feature_rows.len();
                    processed.fetch_add(1, Ordering::Relaxed);
                    total_records_atomic.fetch_add(row_count, Ordering::Relaxed);

                    let processed_count = processed.load(Ordering::Relaxed);
                    if processed_count % 10 == 0 || processed_count <= 100 {
                        println!(
                            "  [{}/{}] {} processed: {} rows (calc: {}ms) - {} total committed",
                            processed_count,
                            total_stocks,
                            ts_code,
                            row_count,
                            calc_elapsed,
                            total_records_atomic.load(Ordering::Relaxed)
                        );
                    }

                    Ok(())
                }
            })
            .buffer_unordered(cli.concurrency);

        // Execute stream and collect errors if any
        let mut stream = stock_stream;
        while let Some(res) = stream.next().await {
            if let Err(e) = res {
                eprintln!("Error processing stock: {:?}", e);
            }
        }

        final_processed = processed.load(Ordering::Relaxed);
        final_skipped = skipped.load(Ordering::Relaxed);
        final_records = total_records_atomic.load(Ordering::Relaxed);

        println!(
            "\n✅ Processed {} stocks, inserted {} records in {:?}",
            final_processed,
            final_records,
            start_time.elapsed()
        );
        if final_skipped > 0 {
            println!(
                "⚠️  Skipped {} stocks due to insufficient historical data (< 60 days)",
                final_skipped
            );
        }
    }

    println!(
        "\n✅ Processed {} stocks, inserted {} records in {:?}",
        final_processed,
        final_records,
        start_time.elapsed()
    );
    if final_skipped > 0 {
        println!(
            "⚠️  Skipped {} stocks due to insufficient historical data (< 60 days)",
            final_skipped
        );
    }

    // Optionally apply monthly winsorization to next_day_return
    if cli.winsorize_monthly {
        println!("⚙️  Applying monthly cross-sectional winsorization (pct={})...", cli.winsorize_pct);
        let updated = apply_monthly_winsorization(&dbpool, cli.winsorize_pct).await?;
        println!("✅ Monthly winsorization applied. Rows updated: {}", updated);
    }

    Ok(())
}

async fn insert_feature_row(
    pool: &Pool<Postgres>,
    row: &FeatureRow,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    sqlx::query(
        r#"
        INSERT INTO ml_training_dataset (
            ts_code, trade_date, industry, act_ent_type, volume, amount, month, weekday, quarter, week_no,
            open_pct, high_pct, low_pct, close_pct, high_from_open_pct, low_from_open_pct, close_from_open_pct,
            intraday_range_pct, close_position_in_range, sma_5, sma_10, sma_20,
            macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
            rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
            asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
            upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20, aroon_up_25,
            return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, volume_roc_5, volume_spike,
            price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, is_shooting_star, consecutive_days,
            index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, index_chinext_pct_chg, index_chinext_vs_ma5_pct,
            index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
            index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
            fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
            net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow,
            turnover_rate, turnover_rate_f, /* volume_ratio, */ pe, pe_ttm, pb, dv_ratio, dv_ttm, total_share, float_share,
            free_share,
            industry_emb_0, industry_emb_1, industry_emb_2, industry_emb_3, industry_emb_4, industry_emb_5, industry_emb_6, industry_emb_7,
            act_ent_type_emb_0, act_ent_type_emb_1, act_ent_type_emb_2, act_ent_type_emb_3, act_ent_type_emb_4, act_ent_type_emb_5, act_ent_type_emb_6, act_ent_type_emb_7,
            close_position_in_range_imputed, macd_monthly_line_imputed, macd_monthly_signal_imputed,
            vol_percentile, high_vol_regime, next_day_return,
            next_day_direction, next_3day_return, next_3day_direction,
            pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days
        ) VALUES (
            -- 1-145: all columns including embeddings and imputation flags
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80,$81,$82,$83,$84,$85,$86,$87,$88,$89,$90,$91,$92,$93,$94,$95,$96,
            $97,$98,$99,$100,$101,$102,$103,$104,$105,$106,$107,$108,$109,$110,$111,$112,$113,$114,$115,$116,$117,$118,$119,$120,$121,$122,$123,$124,$125,$126,$127,$128,$129,$130,$131,$132,$133,$134,$135,$136,$137,$138,$139,$140,$141,$142,$143,$144,$145
        )
        ON CONFLICT (ts_code, trade_date) DO UPDATE SET
            industry_avg_return = COALESCE(ml_training_dataset.industry_avg_return, EXCLUDED.industry_avg_return),
            stock_vs_industry = COALESCE(ml_training_dataset.stock_vs_industry, EXCLUDED.stock_vs_industry),
            industry_momentum_5d = COALESCE(ml_training_dataset.industry_momentum_5d, EXCLUDED.industry_momentum_5d)
    "#, 
    )
    .bind(row.ts_code.clone())
    .bind(row.trade_date.clone())
    .bind(row.industry.clone())
    .bind(row.act_ent_type.clone())
    .bind(row.volume)
    .bind(row.amount)
    .bind(row.month)
    .bind(row.weekday)
    .bind(row.quarter)
    .bind(row.week_no)
    .bind(row.open_pct)
    .bind(row.high_pct)
    .bind(row.low_pct)
    .bind(row.close_pct)
    .bind(row.high_from_open_pct)
    .bind(row.low_from_open_pct)
    .bind(row.close_from_open_pct)
    .bind(row.intraday_range_pct)
    .bind(row.close_position_in_range)
    .bind(row.sma_5)
    .bind(row.sma_10)
    .bind(row.sma_20)
    .bind(row.macd_line)
    .bind(row.macd_signal)
    .bind(row.macd_histogram)
    .bind(row.macd_weekly_line)
    .bind(row.macd_weekly_signal)
    .bind(row.macd_monthly_line)
    .bind(row.macd_monthly_signal)
    .bind(row.rsi_14)
    .bind(row.kdj_k)
    .bind(row.kdj_d)
    .bind(row.kdj_j)
    .bind(row.bb_upper)
    .bind(row.bb_middle)
    .bind(row.bb_lower)
    .bind(row.bb_bandwidth)
    .bind(row.bb_percent_b)
    .bind(row.atr)
    .bind(row.volatility_5)
    .bind(row.volatility_20)
    .bind(row.asi)
    .bind(row.obv)
    .bind(row.volume_ratio)
    .bind(row.price_momentum_5)
    .bind(row.price_momentum_10)
    .bind(row.price_momentum_20)
    .bind(row.price_position_52w)
    .bind(row.body_size)
    .bind(row.upper_shadow)
    .bind(row.lower_shadow)
    .bind(row.trend_strength)
    .bind(row.adx_14)
    .bind(row.vwap_distance_pct)
    .bind(row.cmf_20)
    .bind(row.aroon_up_25)
    .bind(row.return_lag_1)
    .bind(row.return_lag_2)
    .bind(row.return_lag_3)
    .bind(row.overnight_gap)
    .bind(row.gap_pct)
    .bind(row.volume_roc_5)
    .bind(row.volume_spike)
    .bind(row.price_roc_5)
    .bind(row.price_roc_10)
    .bind(row.price_roc_20)
    .bind(row.hist_volatility_20)
    .bind(row.is_doji)
    .bind(row.is_hammer)
    .bind(row.is_shooting_star)
    .bind(row.consecutive_days)
    .bind(row.index_csi300_pct_chg)
    .bind(row.index_csi300_vs_ma5_pct)
    .bind(row.index_csi300_vs_ma20_pct)
    .bind(row.index_chinext_pct_chg)
    .bind(row.index_chinext_vs_ma5_pct)
    .bind(row.index_chinext_vs_ma20_pct)
    .bind(row.index_xin9_pct_chg)
    .bind(row.index_xin9_vs_ma5_pct)
    .bind(row.index_xin9_vs_ma20_pct)
    .bind(row.index_hsi_pct_chg)
    .bind(row.index_hsi_vs_ma5_pct)
    .bind(row.index_hsi_vs_ma20_pct)
    .bind(row.fx_usdcnh_pct_chg)
    .bind(row.fx_usdcnh_vs_ma5_pct)
    .bind(row.fx_usdcnh_vs_ma20_pct)
    .bind(row.net_mf_vol)
    .bind(row.net_mf_amount)
    .bind(row.smart_money_ratio)
    .bind(row.large_order_flow)
    .bind(row.industry_avg_return)
    .bind(row.stock_vs_industry)
    .bind(row.industry_momentum_5d)
    .bind(row.industry_momentum)
    .bind(row.turnover_rate)
    .bind(row.turnover_rate_f)
    .bind(row.pe)
    .bind(row.pe_ttm)
    .bind(row.pb)
    .bind(row.dv_ratio)
    .bind(row.dv_ttm)
    .bind(row.total_share)
    .bind(row.float_share)
    .bind(row.free_share)
    // Embeddings: industry_emb_0..7
    .bind(row.industry_emb[0])
    .bind(row.industry_emb[1])
    .bind(row.industry_emb[2])
    .bind(row.industry_emb[3])
    .bind(row.industry_emb[4])
    .bind(row.industry_emb[5])
    .bind(row.industry_emb[6])
    .bind(row.industry_emb[7])
    // Embeddings: act_ent_type_emb_0..7
    .bind(row.act_ent_type_emb[0])
    .bind(row.act_ent_type_emb[1])
    .bind(row.act_ent_type_emb[2])
    .bind(row.act_ent_type_emb[3])
    .bind(row.act_ent_type_emb[4])
    .bind(row.act_ent_type_emb[5])
    .bind(row.act_ent_type_emb[6])
    .bind(row.act_ent_type_emb[7])
    // Imputation flags
    .bind(row.close_position_in_range_imputed)
    .bind(row.macd_monthly_line_imputed)
    .bind(row.macd_monthly_signal_imputed)
    .bind(row.vol_percentile)
    .bind(row.high_vol_regime)
    .bind(row.next_day_return)
    .bind(row.next_day_direction)
    .bind(row.next_3day_return)
    .bind(row.next_3day_direction)
    .bind(row.pe_percentile_52w)
    .bind(row.sector_momentum_vs_market)
    .bind(row.volume_accel_5d)
    .bind(row.price_vs_52w_high)
    .bind(row.consecutive_up_days)
    .execute(pool)
    .await?;
    Ok(())
}

/// Batch insert multiple feature rows using a single transaction for efficiency
/// Uses smaller chunks (50 rows) and commits immediately for visibility
async fn batch_insert_feature_rows(
    pool: &Pool<Postgres>,
    rows: &[FeatureRow],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if rows.is_empty() {
        return Ok(());
    }

    // Use a transaction for atomic batch insert
    let mut tx = pool.begin().await?;

    // Query existing columns to build dynamic insert that is resilient to column drops
    let cols = sqlx::query!("SELECT column_name FROM information_schema.columns WHERE table_name = 'ml_training_dataset' ORDER BY ordinal_position")
        .fetch_all(pool)
        .await?;
    let existing_cols: std::collections::HashSet<String> = cols.into_iter().filter_map(|r| r.column_name).collect();

    // Desired insertion order (mirrors original hard-coded insert)
    let desired_order: Vec<&str> = vec![
        "ts_code", "trade_date", "industry", "act_ent_type", "volume", "amount", "weekday", "week_no",
        "open_pct", "high_pct", "low_pct", "close_pct", "high_from_open_pct", "low_from_open_pct", "close_from_open_pct",
        "intraday_range_pct", "close_position_in_range", "sma_5",
        "macd_signal", "macd_weekly_signal", "macd_monthly_signal",
        "rsi_14", "kdj_j", "bb_bandwidth", "volatility_5", "volatility_20",
        "asi", "obv", "volume_ratio", "price_momentum_5", "price_momentum_10", "price_momentum_20", "price_position_52w", "body_size",
        "upper_shadow", "lower_shadow", "trend_strength", "vwap_distance_pct", "cmf_20", "aroon_up_25",
        "return_lag_2", "return_lag_3", "volume_roc_5", "volume_spike",
        "is_doji", "is_hammer", "is_shooting_star", "consecutive_days",
        "index_chinext_pct_chg", "index_chinext_vs_ma5_pct", "index_chinext_vs_ma20_pct", "index_xin9_pct_chg", "index_xin9_vs_ma5_pct", "index_xin9_vs_ma20_pct",
        "index_hsi_pct_chg", "index_hsi_vs_ma5_pct", "index_hsi_vs_ma20_pct",
        "fx_usdcnh_vs_ma5_pct", "fx_usdcnh_vs_ma20_pct",
        "net_mf_vol", "net_mf_amount", "smart_money_ratio", "large_order_flow",
        "industry_avg_return", "stock_vs_industry", "industry_momentum_5d",
        "turnover_rate", "pe", "pe_ttm", "pb", "dv_ratio", "dv_ttm", "float_share",
        "free_share",
        // embeddings
        "industry_emb_0","industry_emb_1","industry_emb_2","industry_emb_3","industry_emb_4","industry_emb_5","industry_emb_6","industry_emb_7",
        "act_ent_type_emb_0","act_ent_type_emb_1","act_ent_type_emb_2","act_ent_type_emb_3","act_ent_type_emb_4","act_ent_type_emb_5","act_ent_type_emb_6","act_ent_type_emb_7",
        // imputation flags and others
        "close_position_in_range_imputed", "macd_monthly_line_imputed", "macd_monthly_signal_imputed",
        "vol_percentile", "high_vol_regime", "next_day_return",
        "next_day_direction", "next_3day_return", "next_3day_direction",
        "pe_percentile_52w", "price_vs_52w_high", "consecutive_up_days"
    ];

    let insert_columns: Vec<&str> = desired_order.into_iter().filter(|c| existing_cols.contains(&c.to_string())).collect();

    // Process in chunks of 50 rows for faster visibility (was 500)
    const CHUNK_SIZE: usize = 50;

    for chunk in rows.chunks(CHUNK_SIZE) {
        // Build multi-value INSERT statement dynamically using only existing columns
        let col_list = insert_columns.join(", ");
        let mut sql = format!("INSERT INTO ml_training_dataset ({}) VALUES ", col_list);

        // Generate value placeholders for each row (cols_per_row == insert_columns.len())
        let cols_per_row = insert_columns.len();
        for (row_idx, _row) in chunk.iter().enumerate() {
            if row_idx > 0 {
                sql.push_str(", ");
            }
            sql.push('(');
            for col_idx in 0..cols_per_row {
                if col_idx > 0 {
                    sql.push_str(", ");
                }
                sql.push_str(&format!("${}", row_idx * cols_per_row + col_idx + 1));
            }
            sql.push(')');
        }

        // Upsert: only include update assignments for columns that exist in the current schema
        let mut update_assigns: Vec<String> = Vec::new();
        for &c in insert_columns.iter() {
            if c != "ts_code" && c != "trade_date" {
                update_assigns.push(format!("{} = EXCLUDED.{}", c, c));
            }
        }
        if !update_assigns.is_empty() {
            sql.push_str(&format!(" ON CONFLICT (ts_code, trade_date) DO UPDATE SET {}", update_assigns.join(", ")));
        } else {
            sql.push_str(" ON CONFLICT (ts_code, trade_date) DO NOTHING");
        }

        // Build the query with all bindings
        let mut query = sqlx::query(&sql);

        for row in chunk {
            for col in insert_columns.iter() {
                match *col {
                    "ts_code" => { query = query.bind(&row.ts_code); }
                    "trade_date" => { query = query.bind(&row.trade_date); }
                    "industry" => { query = query.bind(&row.industry); }
                    "act_ent_type" => { query = query.bind(&row.act_ent_type); }
                    "volume" => { query = query.bind(row.volume); }
                    "amount" => { query = query.bind(row.amount); }
                    "month" => { query = query.bind(row.month); }
                    "weekday" => { query = query.bind(row.weekday); }
                    "quarter" => { query = query.bind(row.quarter); }
                    "week_no" => { query = query.bind(row.week_no); }
                    "open_pct" => { query = query.bind(row.open_pct); }
                    "high_pct" => { query = query.bind(row.high_pct); }
                    "low_pct" => { query = query.bind(row.low_pct); }
                    "close_pct" => { query = query.bind(row.close_pct); }
                    "high_from_open_pct" => { query = query.bind(row.high_from_open_pct); }
                    "low_from_open_pct" => { query = query.bind(row.low_from_open_pct); }
                    "close_from_open_pct" => { query = query.bind(row.close_from_open_pct); }
                    "intraday_range_pct" => { query = query.bind(row.intraday_range_pct); }
                    "close_position_in_range" => { query = query.bind(row.close_position_in_range); }
                    "sma_5" => { query = query.bind(row.sma_5); }
                    "sma_10" => { query = query.bind(row.sma_10); }
                    "sma_20" => { query = query.bind(row.sma_20); }
                    "macd_line" => { query = query.bind(row.macd_line); }
                    "macd_signal" => { query = query.bind(row.macd_signal); }
                    "macd_histogram" => { query = query.bind(row.macd_histogram); }
                    "macd_weekly_line" => { query = query.bind(row.macd_weekly_line); }
                    "macd_weekly_signal" => { query = query.bind(row.macd_weekly_signal); }
                    "macd_monthly_line" => { query = query.bind(row.macd_monthly_line); }
                    "macd_monthly_signal" => { query = query.bind(row.macd_monthly_signal); }
                    "rsi_14" => { query = query.bind(row.rsi_14); }
                    "kdj_k" => { query = query.bind(row.kdj_k); }
                    "kdj_d" => { query = query.bind(row.kdj_d); }
                    "kdj_j" => { query = query.bind(row.kdj_j); }
                    "bb_upper" => { query = query.bind(row.bb_upper); }
                    "bb_middle" => { query = query.bind(row.bb_middle); }
                    "bb_lower" => { query = query.bind(row.bb_lower); }
                    "bb_bandwidth" => { query = query.bind(row.bb_bandwidth); }
                    "bb_percent_b" => { query = query.bind(row.bb_percent_b); }
                    "atr" => { query = query.bind(row.atr); }
                    "volatility_5" => { query = query.bind(row.volatility_5); }
                    "volatility_20" => { query = query.bind(row.volatility_20); }
                    "asi" => { query = query.bind(row.asi); }
                    "obv" => { query = query.bind(row.obv); }
                    "volume_ratio" => { query = query.bind(row.volume_ratio); }
                    "price_momentum_5" => { query = query.bind(row.price_momentum_5); }
                    "price_momentum_10" => { query = query.bind(row.price_momentum_10); }
                    "price_momentum_20" => { query = query.bind(row.price_momentum_20); }
                    "price_position_52w" => { query = query.bind(row.price_position_52w); }
                    "body_size" => { query = query.bind(row.body_size); }
                    "upper_shadow" => { query = query.bind(row.upper_shadow); }
                    "lower_shadow" => { query = query.bind(row.lower_shadow); }
                    "trend_strength" => { query = query.bind(row.trend_strength); }
                    "adx_14" => { query = query.bind(row.adx_14); }
                    "vwap_distance_pct" => { query = query.bind(row.vwap_distance_pct); }
                    "cmf_20" => { query = query.bind(row.cmf_20); }
                    "aroon_up_25" => { query = query.bind(row.aroon_up_25); }
                    "return_lag_1" => { query = query.bind(row.return_lag_1); }
                    "return_lag_2" => { query = query.bind(row.return_lag_2); }
                    "return_lag_3" => { query = query.bind(row.return_lag_3); }
                    "overnight_gap" => { query = query.bind(row.overnight_gap); }
                    "gap_pct" => { query = query.bind(row.gap_pct); }
                    "volume_roc_5" => { query = query.bind(row.volume_roc_5); }
                    "volume_spike" => { query = query.bind(row.volume_spike); }
                    "price_roc_5" => { query = query.bind(row.price_roc_5); }
                    "price_roc_10" => { query = query.bind(row.price_roc_10); }
                    "price_roc_20" => { query = query.bind(row.price_roc_20); }
                    "hist_volatility_20" => { query = query.bind(row.hist_volatility_20); }
                    "is_doji" => { query = query.bind(row.is_doji); }
                    "is_hammer" => { query = query.bind(row.is_hammer); }
                    "is_shooting_star" => { query = query.bind(row.is_shooting_star); }
                    "consecutive_days" => { query = query.bind(row.consecutive_days); }
                    "index_csi300_pct_chg" => { query = query.bind(row.index_csi300_pct_chg); }
                    "index_csi300_vs_ma5_pct" => { query = query.bind(row.index_csi300_vs_ma5_pct); }
                    "index_csi300_vs_ma20_pct" => { query = query.bind(row.index_csi300_vs_ma20_pct); }
                    "index_chinext_pct_chg" => { query = query.bind(row.index_chinext_pct_chg); }
                    "index_chinext_vs_ma5_pct" => { query = query.bind(row.index_chinext_vs_ma5_pct); }
                    "index_chinext_vs_ma20_pct" => { query = query.bind(row.index_chinext_vs_ma20_pct); }
                    "index_xin9_pct_chg" => { query = query.bind(row.index_xin9_pct_chg); }
                    "index_xin9_vs_ma5_pct" => { query = query.bind(row.index_xin9_vs_ma5_pct); }
                    "index_xin9_vs_ma20_pct" => { query = query.bind(row.index_xin9_vs_ma20_pct); }
                    "index_hsi_pct_chg" => { query = query.bind(row.index_hsi_pct_chg); }
                    "index_hsi_vs_ma5_pct" => { query = query.bind(row.index_hsi_vs_ma5_pct); }
                    "index_hsi_vs_ma20_pct" => { query = query.bind(row.index_hsi_vs_ma20_pct); }
                    "fx_usdcnh_pct_chg" => { query = query.bind(row.fx_usdcnh_pct_chg); }
                    "fx_usdcnh_vs_ma5_pct" => { query = query.bind(row.fx_usdcnh_vs_ma5_pct); }
                    "fx_usdcnh_vs_ma20_pct" => { query = query.bind(row.fx_usdcnh_vs_ma20_pct); }
                    "net_mf_vol" => { query = query.bind(row.net_mf_vol); }
                    "net_mf_amount" => { query = query.bind(row.net_mf_amount); }
                    "smart_money_ratio" => { query = query.bind(row.smart_money_ratio); }
                    "large_order_flow" => { query = query.bind(row.large_order_flow); }
                    "industry_avg_return" => { query = query.bind(row.industry_avg_return); }
                    "stock_vs_industry" => { query = query.bind(row.stock_vs_industry); }
                    "industry_momentum_5d" => { query = query.bind(row.industry_momentum_5d); }
                    "industry_momentum" => { query = query.bind(row.industry_momentum); }
                    "turnover_rate" => { query = query.bind(row.turnover_rate); }
                    "turnover_rate_f" => { query = query.bind(row.turnover_rate_f); }
                    "pe" => { query = query.bind(row.pe); }
                    "pe_ttm" => { query = query.bind(row.pe_ttm); }
                    "pb" => { query = query.bind(row.pb); }
                    "dv_ratio" => { query = query.bind(row.dv_ratio); }
                    "dv_ttm" => { query = query.bind(row.dv_ttm); }
                    "total_share" => { query = query.bind(row.total_share); }
                    "float_share" => { query = query.bind(row.float_share); }
                    "free_share" => { query = query.bind(row.free_share); }
                    "industry_emb_0" => { query = query.bind(row.industry_emb[0]); }
                    "industry_emb_1" => { query = query.bind(row.industry_emb[1]); }
                    "industry_emb_2" => { query = query.bind(row.industry_emb[2]); }
                    "industry_emb_3" => { query = query.bind(row.industry_emb[3]); }
                    "industry_emb_4" => { query = query.bind(row.industry_emb[4]); }
                    "industry_emb_5" => { query = query.bind(row.industry_emb[5]); }
                    "industry_emb_6" => { query = query.bind(row.industry_emb[6]); }
                    "industry_emb_7" => { query = query.bind(row.industry_emb[7]); }
                    "act_ent_type_emb_0" => { query = query.bind(row.act_ent_type_emb[0]); }
                    "act_ent_type_emb_1" => { query = query.bind(row.act_ent_type_emb[1]); }
                    "act_ent_type_emb_2" => { query = query.bind(row.act_ent_type_emb[2]); }
                    "act_ent_type_emb_3" => { query = query.bind(row.act_ent_type_emb[3]); }
                    "act_ent_type_emb_4" => { query = query.bind(row.act_ent_type_emb[4]); }
                    "act_ent_type_emb_5" => { query = query.bind(row.act_ent_type_emb[5]); }
                    "act_ent_type_emb_6" => { query = query.bind(row.act_ent_type_emb[6]); }
                    "act_ent_type_emb_7" => { query = query.bind(row.act_ent_type_emb[7]); }
                    "close_position_in_range_imputed" => { query = query.bind(row.close_position_in_range_imputed); }
                    "macd_monthly_line_imputed" => { query = query.bind(row.macd_monthly_line_imputed); }
                    "macd_monthly_signal_imputed" => { query = query.bind(row.macd_monthly_signal_imputed); }
                    "vol_percentile" => { query = query.bind(row.vol_percentile); }
                    "high_vol_regime" => { query = query.bind(row.high_vol_regime); }
                    "next_day_return" => { query = query.bind(row.next_day_return); }
                    "next_day_direction" => { query = query.bind(row.next_day_direction); }
                    "next_3day_return" => { query = query.bind(row.next_3day_return); }
                    "next_3day_direction" => { query = query.bind(row.next_3day_direction); }
                    "pe_percentile_52w" => { query = query.bind(row.pe_percentile_52w); }
                    "sector_momentum_vs_market" => { query = query.bind(row.sector_momentum_vs_market); }
                    "volume_accel_5d" => { query = query.bind(row.volume_accel_5d); }
                    "price_vs_52w_high" => { query = query.bind(row.price_vs_52w_high); }
                    "consecutive_up_days" => { query = query.bind(row.consecutive_up_days); }
                    _ => { /* unknown/removed column; skip binding */ }
                }
            }
        }

        query.execute(&mut *tx).await?;
    }

    tx.commit().await?;
    Ok(())
}

/// Helper struct for index daily data
#[derive(Clone, Debug)]
struct IndexDaily {
    trade_date: String,
    close: f64,
    pct_chg: f64,
    ma5: Option<f64>,
    ma20: Option<f64>,
}

// --- Fix: Use subqueries to calculate ma5/ma20 on the fly, since index_daily does not have ma5/ma20 columns ---
async fn prefetch_index_data(
    pool: &Pool<Postgres>,
    index_codes: &[&str],
    min_date: &str,
    max_date: &str,
) -> HashMap<(String, String), IndexDaily> {
    let mut map = HashMap::new();
    for &code in index_codes {
        let code_str = code.to_string();

        // --- DEBUG: Print which index code is being loaded ---
        println!("[DEBUG] Loading index data for code: {}", code);

        let rows = sqlx::query!(
            r#"
            SELECT 
                COALESCE(ts_code, '') as "ts_code!",
                COALESCE(trade_date, '') as "trade_date!",
                COALESCE(close::DOUBLE PRECISION, 0.0) as "close!",
                COALESCE(pct_chg::DOUBLE PRECISION, 0.0) as "pct_chg!",
                AVG(close::DOUBLE PRECISION) OVER (
                    PARTITION BY ts_code 
                    ORDER BY trade_date 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS ma5,
                AVG(close::DOUBLE PRECISION) OVER (
                    PARTITION BY ts_code 
                    ORDER BY trade_date 
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS ma20
            FROM index_daily
            WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
            ORDER BY trade_date
            "#,
            code,
            min_date,
            max_date
        )
        .fetch_all(pool)
        .await
        .unwrap_or_else(|e| {
            println!(
                "[DEBUG] Failed to fetch index_daily for code {}: {}",
                code, e
            );
            Vec::new()
        });

        // --- DEBUG: Print how many rows were loaded for this code ---
        println!(
            "[DEBUG] Loaded {} rows for index code {} (date range {} to {})",
            rows.len(),
            code,
            min_date,
            max_date
        );

        // --- DEBUG: Print first 3 and last 3 rows for this code ---
        for (i, row) in rows.iter().take(3).enumerate() {
            println!(
                "[DEBUG] {}: ts_code={}, trade_date={}, close={:?}, pct_chg={:?}, ma5={:?}, ma20={:?}",
                i, row.ts_code, row.trade_date, row.close, row.pct_chg, row.ma5, row.ma20
            );
        }
        if rows.len() > 3 {
            for (i, row) in rows.iter().rev().take(3).enumerate() {
                println!(
                    "[DEBUG] -{}: ts_code={}, trade_date={}, close={:?}, pct_chg={:?}, ma5={:?}, ma20={:?}",
                    i + 1,
                    row.ts_code,
                    row.trade_date,
                    row.close,
                    row.pct_chg,
                    row.ma5,
                    row.ma20
                );
            }
        }

        for row in rows {
            map.insert(
                (code_str.clone(), row.trade_date.clone()),
                IndexDaily {
                    trade_date: row.trade_date,
                    close: row.close,
                    pct_chg: row.pct_chg,
                    ma5: row.ma5,
                    ma20: row.ma20,
                },
            );
        }
    }
    // --- DEBUG: Print total keys loaded ---
    println!("[DEBUG] Total index_data keys loaded: {}", map.len());
    map
}

/// Pre-fetch industry performance (average daily return and 5-day industry momentum)
async fn prefetch_industry_performance(
    pool: &Pool<Postgres>,
    min_date: &str,
    max_date: &str,
) -> std::collections::HashMap<String, Vec<(String, (f64, f64))>> {
    // Compute per-industry daily average return then compute 5-day rolling average per industry
    let rows = sqlx::query!(
        r#"
        WITH daily_industry AS (
            SELECT sb.industry, sd.trade_date, AVG(sd.pct_chg::DOUBLE PRECISION) AS avg_return
            FROM stock_daily sd
            JOIN stock_basic sb ON sd.ts_code = sb.ts_code
            WHERE sd.trade_date >= $1 AND sd.trade_date <= $2 AND sb.industry IS NOT NULL
            GROUP BY sb.industry, sd.trade_date
            ORDER BY sb.industry, sd.trade_date
        )
        SELECT industry, trade_date, avg_return,
            AVG(avg_return) OVER (PARTITION BY industry ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS avg_5d
        FROM daily_industry
        "#,
        min_date,
        max_date
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut map: std::collections::HashMap<String, Vec<(String, (f64, f64))>> = std::collections::HashMap::new();
    for r in rows {
        let ind = r.industry.clone().unwrap_or_else(|| "UNKNOWN".to_string());
        let trade_date = r.trade_date.clone();
        let vals = (r.avg_return.unwrap_or(0.0), r.avg_5d.unwrap_or(0.0));
        map.entry(ind).or_insert_with(Vec::new).push((trade_date, vals));
    }

    // Ensure each industry's vector is sorted by trade_date ascending for binary search
    for (_ind, vec) in map.iter_mut() {
        vec.sort_by(|a, b| a.0.cmp(&b.0));
    }

    // --- DEBUG: Print total keys loaded ---
    println!("[DEBUG] Total industries loaded: {}", map.len());
    map
} 

fn hash_to_embedding_fixed(s: &str, dim: usize, seed: u64) -> Vec<f64> {
    use std::hash::{Hasher, Hash};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    let mut state = hasher.finish().wrapping_add(seed);
    let mut out: Vec<f64> = Vec::with_capacity(dim);
    for _ in 0..dim {
        // splitmix64 inspired
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        let v = (z as f64) / (u64::MAX as f64);
        out.push(v * 2.0 - 1.0);
    }
    // normalize
    let norm: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

fn get_ml_create_table_sql() -> &'static str {
    r#"
    CREATE TABLE IF NOT EXISTS ml_training_dataset (
        ts_code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        PRIMARY KEY (ts_code, trade_date),
        industry TEXT,
        act_ent_type TEXT,
        volume DOUBLE PRECISION,
        amount DOUBLE PRECISION,
        month SMALLINT,
        weekday SMALLINT,
        quarter SMALLINT,
        week_no SMALLINT,
        open_pct DOUBLE PRECISION,
        high_pct DOUBLE PRECISION,
        low_pct DOUBLE PRECISION,
        close_pct DOUBLE PRECISION,
        high_from_open_pct DOUBLE PRECISION,
        low_from_open_pct DOUBLE PRECISION,
        close_from_open_pct DOUBLE PRECISION,
        intraday_range_pct DOUBLE PRECISION,
        close_position_in_range DOUBLE PRECISION,
        sma_5 DOUBLE PRECISION,
        sma_10 DOUBLE PRECISION,
        sma_20 DOUBLE PRECISION,
        macd_line DOUBLE PRECISION,
        macd_signal DOUBLE PRECISION,
        macd_histogram DOUBLE PRECISION,
        macd_weekly_line DOUBLE PRECISION,
        macd_weekly_signal DOUBLE PRECISION,
        macd_monthly_line DOUBLE PRECISION,
        macd_monthly_signal DOUBLE PRECISION,
        rsi_14 DOUBLE PRECISION,
        kdj_k DOUBLE PRECISION,
        kdj_d DOUBLE PRECISION,
        kdj_j DOUBLE PRECISION,
        bb_upper DOUBLE PRECISION,
        bb_middle DOUBLE PRECISION,
        bb_lower DOUBLE PRECISION,
        bb_bandwidth DOUBLE PRECISION,
        bb_percent_b DOUBLE PRECISION,
        atr DOUBLE PRECISION,
        volatility_5 DOUBLE PRECISION,
        volatility_20 DOUBLE PRECISION,
        asi DOUBLE PRECISION,
        obv DOUBLE PRECISION,
        volume_ratio DOUBLE PRECISION,
        price_momentum_5 DOUBLE PRECISION,
        price_momentum_10 DOUBLE PRECISION,
        price_momentum_20 DOUBLE PRECISION,
        price_position_52w DOUBLE PRECISION,
        body_size DOUBLE PRECISION,
        upper_shadow DOUBLE PRECISION,
        lower_shadow DOUBLE PRECISION,
        trend_strength DOUBLE PRECISION,
        adx_14 DOUBLE PRECISION,
        vwap_distance_pct DOUBLE PRECISION,
        cmf_20 DOUBLE PRECISION,
        aroon_up_25 DOUBLE PRECISION,
        return_lag_1 DOUBLE PRECISION,
        return_lag_2 DOUBLE PRECISION,
        return_lag_3 DOUBLE PRECISION,
        overnight_gap DOUBLE PRECISION,
        gap_pct DOUBLE PRECISION,
        volume_roc_5 DOUBLE PRECISION,
        volume_spike BOOLEAN,
        price_roc_5 DOUBLE PRECISION,
        price_roc_10 DOUBLE PRECISION,
        price_roc_20 DOUBLE PRECISION,
        hist_volatility_20 DOUBLE PRECISION,
        is_doji BOOLEAN,
        is_hammer BOOLEAN,
        is_shooting_star BOOLEAN,
        consecutive_days INTEGER,
        index_csi300_pct_chg DOUBLE PRECISION,
        index_csi300_vs_ma5_pct DOUBLE PRECISION,
        index_csi300_vs_ma20_pct DOUBLE PRECISION,
        index_chinext_pct_chg DOUBLE PRECISION,
        index_chinext_vs_ma5_pct DOUBLE PRECISION,
        index_chinext_vs_ma20_pct DOUBLE PRECISION,
        index_xin9_pct_chg DOUBLE PRECISION,
        index_xin9_vs_ma5_pct DOUBLE PRECISION,
        index_xin9_vs_ma20_pct DOUBLE PRECISION,
        index_hsi_pct_chg DOUBLE PRECISION,
        index_hsi_vs_ma5_pct DOUBLE PRECISION,
        index_hsi_vs_ma20_pct DOUBLE PRECISION,
        -- DailyBasic columns (excluding close)
        turnover_rate DOUBLE PRECISION,
        turnover_rate_f DOUBLE PRECISION,
        pe DOUBLE PRECISION,
        pe_ttm DOUBLE PRECISION,
        pb DOUBLE PRECISION,
        dv_ratio DOUBLE PRECISION,
        dv_ttm DOUBLE PRECISION,
        total_share DOUBLE PRECISION,
        float_share DOUBLE PRECISION,
        free_share DOUBLE PRECISION,
        -- Embeddings for categorical features (industry, act_ent_type) - 8 dims each
        industry_emb_0 DOUBLE PRECISION,
        industry_emb_1 DOUBLE PRECISION,
        industry_emb_2 DOUBLE PRECISION,
        industry_emb_3 DOUBLE PRECISION,
        industry_emb_4 DOUBLE PRECISION,
        industry_emb_5 DOUBLE PRECISION,
        industry_emb_6 DOUBLE PRECISION,
        industry_emb_7 DOUBLE PRECISION,
        act_ent_type_emb_0 DOUBLE PRECISION,
        act_ent_type_emb_1 DOUBLE PRECISION,
        act_ent_type_emb_2 DOUBLE PRECISION,
        act_ent_type_emb_3 DOUBLE PRECISION,
        act_ent_type_emb_4 DOUBLE PRECISION,
        act_ent_type_emb_5 DOUBLE PRECISION,
        act_ent_type_emb_6 DOUBLE PRECISION,
        act_ent_type_emb_7 DOUBLE PRECISION,
        -- Imputation flags for fields that were sometimes missing
        close_position_in_range_imputed BOOLEAN,
        macd_monthly_line_imputed BOOLEAN,
        macd_monthly_signal_imputed BOOLEAN,
        vol_percentile DOUBLE PRECISION,
        high_vol_regime SMALLINT,
        next_day_return DOUBLE PRECISION,
        next_day_direction SMALLINT,
        next_3day_return DOUBLE PRECISION,
        next_3day_direction SMALLINT
    );
    "#
}

// --- Add these stubs near the top of your file ---

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}

fn pearson_corr(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if den_a == 0.0 || den_b == 0.0 {
        return None;
    }
    Some(num / den_a.sqrt() / den_b.sqrt())
}

fn invert_matrix(mut m: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    // build augmented matrix [m | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        if m[i].len() != n { return None; }
        for j in 0..n { aug[i][j] = m[i][j]; }
        aug[i][n + i] = 1.0;
    }
    // Gauss-Jordan elimination
    for i in 0..n {
        // find pivot
        let mut pivot = i;
        for r in i..n {
            if aug[r][i].abs() > aug[pivot][i].abs() { pivot = r; }
        }
        if aug[pivot][i].abs() < 1e-12 { return None; } // singular
        if pivot != i { aug.swap(i, pivot); }
        let diag = aug[i][i];
        for c in 0..2*n { aug[i][c] /= diag; }
        for r in 0..n {
            if r == i { continue; }
            let factor = aug[r][i];
            if factor.abs() < 1e-15 { continue; }
            for c in i..2*n {
                aug[r][c] -= factor * aug[i][c];
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n { inv[i][j] = aug[i][n + j]; }
    }
    Some(inv)
}

async fn compute_vif_and_drop_columns(pool: &Pool<Postgres>, rows: &[FeatureRow], execute_drops: bool) -> Result<(), Box<dyn Error + Send + Sync>> {
    use std::fs::File;
    use std::io::Write;

    if rows.is_empty() {
        println!("No rows to compute VIF on.");
        return Ok(())
    }

    // Numeric feature list (exclude embeddings, labels, booleans, and IDs)
    const NUMERIC_COLS: &[&str] = &[
        "volume","amount","open_pct","high_pct","low_pct","close_pct","high_from_open_pct","low_from_open_pct","close_from_open_pct",
        "intraday_range_pct","close_position_in_range","sma_5","sma_10","sma_20",
        "macd_line","macd_signal","macd_histogram","macd_weekly_line","macd_weekly_signal","macd_monthly_line","macd_monthly_signal",
        "rsi_14","kdj_k","kdj_d","kdj_j","bb_upper","bb_middle","bb_lower","bb_bandwidth","bb_percent_b","atr","volatility_5","volatility_20",
        "asi","obv","volume_ratio","price_momentum_5","price_momentum_10","price_momentum_20","price_position_52w","body_size","upper_shadow","lower_shadow",
        "trend_strength","adx_14","vwap_distance_pct","cmf_20","aroon_up_25","return_lag_1","return_lag_2","return_lag_3",
        "overnight_gap","gap_pct","volume_roc_5","price_roc_5","price_roc_10","price_roc_20","hist_volatility_20","consecutive_days",
        "index_csi300_pct_chg","index_csi300_vs_ma5_pct","index_csi300_vs_ma20_pct","index_chinext_pct_chg","index_chinext_vs_ma5_pct","index_chinext_vs_ma20_pct",
        "index_xin9_pct_chg","index_xin9_vs_ma5_pct","index_xin9_vs_ma20_pct","index_hsi_pct_chg","index_hsi_vs_ma5_pct","index_hsi_vs_ma20_pct",
        "fx_usdcnh_pct_chg","fx_usdcnh_vs_ma5_pct","fx_usdcnh_vs_ma20_pct","net_mf_vol","net_mf_amount","smart_money_ratio","large_order_flow",
        "industry_avg_return","stock_vs_industry","industry_momentum_5d","industry_momentum","turnover_rate","turnover_rate_f","pe","pe_ttm","pb",
        "dv_ratio","dv_ttm","total_share","float_share","free_share","vol_percentile","pe_percentile_52w","sector_momentum_vs_market",
        "volume_accel_5d","price_vs_52w_high","consecutive_up_days"
    ];

    let n_features = NUMERIC_COLS.len();
    let n_rows = rows.len();

    // Build column vectors with Option<f64> -> impute median later
    let mut data: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rows); n_features];

    for r in rows {
        for (i, col) in NUMERIC_COLS.iter().enumerate() {
            let v = match *col {
                "volume" => Some(r.volume),
                "amount" => r.amount,
                "open_pct" => r.open_pct,
                "high_pct" => r.high_pct,
                "low_pct" => r.low_pct,
                "close_pct" => r.close_pct,
                "high_from_open_pct" => r.high_from_open_pct,
                "low_from_open_pct" => r.low_from_open_pct,
                "close_from_open_pct" => r.close_from_open_pct,
                "intraday_range_pct" => r.intraday_range_pct,
                "close_position_in_range" => r.close_position_in_range,
                "sma_5" => r.sma_5,
                "sma_10" => r.sma_10,
                "sma_20" => r.sma_20,
                "macd_line" => r.macd_line,
                "macd_signal" => r.macd_signal,
                "macd_histogram" => r.macd_histogram,
                "macd_weekly_line" => r.macd_weekly_line,
                "macd_weekly_signal" => r.macd_weekly_signal,
                "macd_monthly_line" => r.macd_monthly_line,
                "macd_monthly_signal" => r.macd_monthly_signal,
                "rsi_14" => r.rsi_14,
                "kdj_k" => r.kdj_k,
                "kdj_d" => r.kdj_d,
                "kdj_j" => r.kdj_j,
                "bb_upper" => r.bb_upper,
                "bb_middle" => r.bb_middle,
                "bb_lower" => r.bb_lower,
                "bb_bandwidth" => r.bb_bandwidth,
                "bb_percent_b" => r.bb_percent_b,
                "atr" => r.atr,
                "volatility_5" => r.volatility_5,
                "volatility_20" => r.volatility_20,
                "asi" => r.asi,
                "obv" => r.obv,
                "volume_ratio" => r.volume_ratio,
                "price_momentum_5" => r.price_momentum_5,
                "price_momentum_10" => r.price_momentum_10,
                "price_momentum_20" => r.price_momentum_20,
                "price_position_52w" => r.price_position_52w,
                "body_size" => r.body_size,
                "upper_shadow" => r.upper_shadow,
                "lower_shadow" => r.lower_shadow,
                "trend_strength" => r.trend_strength,
                "adx_14" => r.adx_14,
                "vwap_distance_pct" => r.vwap_distance_pct,
                "cmf_20" => r.cmf_20,
                "aroon_up_25" => r.aroon_up_25,
                "return_lag_1" => r.return_lag_1,
                "return_lag_2" => r.return_lag_2,
                "return_lag_3" => r.return_lag_3,
                "overnight_gap" => r.overnight_gap,
                "gap_pct" => r.gap_pct,
                "volume_roc_5" => r.volume_roc_5,
                "price_roc_5" => r.price_roc_5,
                "price_roc_10" => r.price_roc_10,
                "price_roc_20" => r.price_roc_20,
                "hist_volatility_20" => r.hist_volatility_20,
                "consecutive_days" => r.consecutive_days.map(|v| v as f64),
                "index_csi300_pct_chg" => r.index_csi300_pct_chg,
                "index_csi300_vs_ma5_pct" => r.index_csi300_vs_ma5_pct,
                "index_csi300_vs_ma20_pct" => r.index_csi300_vs_ma20_pct,
                "index_chinext_pct_chg" => r.index_chinext_pct_chg,
                "index_chinext_vs_ma5_pct" => r.index_chinext_vs_ma5_pct,
                "index_chinext_vs_ma20_pct" => r.index_chinext_vs_ma20_pct,
                "index_xin9_pct_chg" => r.index_xin9_pct_chg,
                "index_xin9_vs_ma5_pct" => r.index_xin9_vs_ma5_pct,
                "index_xin9_vs_ma20_pct" => r.index_xin9_vs_ma20_pct,
                "index_hsi_pct_chg" => r.index_hsi_pct_chg,
                "index_hsi_vs_ma5_pct" => r.index_hsi_vs_ma5_pct,
                "index_hsi_vs_ma20_pct" => r.index_hsi_vs_ma20_pct,
                "fx_usdcnh_pct_chg" => r.fx_usdcnh_pct_chg,
                "fx_usdcnh_vs_ma5_pct" => r.fx_usdcnh_vs_ma5_pct,
                "fx_usdcnh_vs_ma20_pct" => r.fx_usdcnh_vs_ma20_pct,
                "net_mf_vol" => r.net_mf_vol,
                "net_mf_amount" => r.net_mf_amount,
                "smart_money_ratio" => r.smart_money_ratio,
                "large_order_flow" => r.large_order_flow,
                "industry_avg_return" => r.industry_avg_return,
                "stock_vs_industry" => r.stock_vs_industry,
                "industry_momentum_5d" => r.industry_momentum_5d,
                "industry_momentum" => r.industry_momentum,
                "turnover_rate" => r.turnover_rate,
                "turnover_rate_f" => r.turnover_rate_f,
                "pe" => r.pe,
                "pe_ttm" => r.pe_ttm,
                "pb" => r.pb,
                "dv_ratio" => r.dv_ratio,
                "dv_ttm" => r.dv_ttm,
                "total_share" => r.total_share,
                "float_share" => r.float_share,
                "free_share" => r.free_share,
                "vol_percentile" => r.vol_percentile,
                "pe_percentile_52w" => r.pe_percentile_52w,
                "sector_momentum_vs_market" => r.sector_momentum_vs_market,
                "volume_accel_5d" => r.volume_accel_5d,
                "price_vs_52w_high" => r.price_vs_52w_high,
                "consecutive_up_days" => r.consecutive_up_days.map(|v| v as f64),
                _ => None,
            };
            data[i].push(v.unwrap_or(f64::NAN));
        }
    }

    // Impute medians for NaNs
    for col in data.iter_mut() {
        let mut vals: Vec<f64> = col.iter().cloned().filter(|x| x.is_finite()).collect();
        let med = median(&mut vals);
        for x in col.iter_mut() { if !x.is_finite() { *x = med; } }
    }

    // If user requested log1p/standardize, we will expose helpers to transform the dataset


    // Build correlation matrix
    let mut cor: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n_features];
    for i in 0..n_features {
        for j in i..n_features {
            if i == j { cor[i][j] = 1.0; continue; }
            let c = pearson_corr(&data[i], &data[j]).unwrap_or(0.0);
            cor[i][j] = c;
            cor[j][i] = c;
        }
    }

    // Invert correlation matrix
    let inv = match invert_matrix(cor.clone()) {
        Some(m) => m,
        None => {
            eprintln!("⚠️  Correlation matrix singular; cannot invert for VIF. No columns dropped.");
            // Save report and return
            let mut f = File::create("artifacts/vif_report.csv")?;
            writeln!(f, "feature,vif")?;
            return Ok(());
        }
    };

    // Compute VIFs as diagonal elements
    let mut vifs: Vec<(String, f64)> = Vec::with_capacity(n_features);
    for i in 0..n_features {
        let vif = inv[i][i];
        vifs.push((NUMERIC_COLS[i].to_string(), vif));
    }

    // Write VIF report
    std::fs::create_dir_all("artifacts")?;
    let mut f = File::create("artifacts/vif_report.csv")?;
    writeln!(f, "feature,vif")?;
    for (feat, v) in vifs.iter() {
        writeln!(f, "{},{}", feat, v)?;
    }

    // Identify features to drop (VIF > 10 or non-finite)
    let drop_feats: Vec<&String> = vifs.iter().filter(|(_f,v)| !v.is_finite() || *v > 10.0).map(|(f,_v)| f).collect();
    if drop_feats.is_empty() {
        println!("✅ No high-VIF features detected (all <= 10). Report saved to artifacts/vif_report.csv");
        return Ok(());
    }

    println!("⚠️  Dropping {} high-VIF features: {:?}", drop_feats.len(), drop_feats);
    // Create SQL file with ALTERs
    let mut sql_file = File::create("artifacts/vif_drop.sql")?;
    for fdrop in drop_feats.iter() {
        let stmt = format!("ALTER TABLE ml_training_dataset DROP COLUMN IF EXISTS {};", fdrop);
        writeln!(sql_file, "{}", stmt)?;
        // Execute drop only when requested (avoid schema changes during dry-run)
        if execute_drops {
            sqlx::query(&stmt).execute(pool).await.ok();
        }
    }
    println!("✅ Dropped high-VIF columns and saved ALTER statements to artifacts/vif_drop.sql");
    Ok(())
}

// Apply log1p and/or standard scaling to a collected set of FeatureRows in-place.
// - `apply_log1p`: apply log1p to selected positive-skewed features (only if > 0)
// - `standardize`: z-score numeric features across the dataset (after median imputation and optional log1p)
fn apply_log1p_and_standardize(rows: &mut [FeatureRow], apply_log1p: bool, standardize: bool) -> Result<(), Box<dyn Error + Send + Sync>> {
    use std::fs::File;
    use std::io::Write;

    if rows.is_empty() {
        // No collected rows (e.g., transforms requested but rows were inserted directly per-stock).
        // Still create artifacts so downstream tooling can reliably find parameter files.
        std::fs::create_dir_all("artifacts")?;
        // Write an (empty) winsor params file with header for reproducibility / inspection
        let mut wf = File::create("artifacts/winsor_params.csv")?;
        writeln!(wf, "feature,lo,hi")?;
        // If standardization was requested, produce an empty scale params header so users can inspect structure
        if standardize {
            let mut f = File::create("artifacts/scale_params.csv")?;
            writeln!(f, "feature,applied_log1p,mean,std")?;
        }

        return Ok(());
    }

    const NUMERIC_COLS: &[&str] = &[
        "volume","amount","open_pct","high_pct","low_pct","close_pct","high_from_open_pct","low_from_open_pct","close_from_open_pct",
        "intraday_range_pct","close_position_in_range","sma_5","sma_10","sma_20",
        "macd_line","macd_signal","macd_histogram","macd_weekly_line","macd_weekly_signal","macd_monthly_line","macd_monthly_signal",
        "rsi_14","kdj_k","kdj_d","kdj_j","bb_upper","bb_middle","bb_lower","bb_bandwidth","bb_percent_b","atr","volatility_5","volatility_20",
        "asi","obv","volume_ratio","price_momentum_5","price_momentum_10","price_momentum_20","price_position_52w","body_size","upper_shadow","lower_shadow",
        "trend_strength","adx_14","vwap_distance_pct","cmf_20","aroon_up_25","return_lag_1","return_lag_2","return_lag_3",
        "overnight_gap","gap_pct","volume_roc_5","price_roc_5","price_roc_10","price_roc_20","hist_volatility_20","consecutive_days",
        "index_csi300_pct_chg","index_csi300_vs_ma5_pct","index_csi300_vs_ma20_pct","index_chinext_pct_chg","index_chinext_vs_ma5_pct","index_chinext_vs_ma20_pct",
        "index_xin9_pct_chg","index_xin9_vs_ma5_pct","index_xin9_vs_ma20_pct","index_hsi_pct_chg","index_hsi_vs_ma5_pct","index_hsi_vs_ma20_pct",
        "fx_usdcnh_pct_chg","fx_usdcnh_vs_ma5_pct","fx_usdcnh_vs_ma20_pct","net_mf_vol","net_mf_amount","smart_money_ratio","large_order_flow",
        "industry_avg_return","stock_vs_industry","industry_momentum_5d","industry_momentum","turnover_rate","turnover_rate_f","pe","pe_ttm","pb",
        "dv_ratio","dv_ttm","total_share","float_share","free_share","vol_percentile","pe_percentile_52w","sector_momentum_vs_market",
        "volume_accel_5d","price_vs_52w_high","consecutive_up_days"
    ];

    // conservative set of positive-skewed features to apply log1p to
    const LOG1P_COLS: &[&str] = &[
        "volume","amount","turnover_rate","total_share","float_share","free_share",
        "net_mf_vol","net_mf_amount","large_order_flow","volume_accel_5d","volatility_20","atr","pe","pe_ttm","pb"
    ];

    let n_features = NUMERIC_COLS.len();
    let n_rows = rows.len();

    let mut data: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rows); n_features];

    for r in rows.iter() {
        for (i, col) in NUMERIC_COLS.iter().enumerate() {
            let v = match *col {
                "volume" => Some(r.volume),
                "amount" => r.amount,
                "open_pct" => r.open_pct,
                "high_pct" => r.high_pct,
                "low_pct" => r.low_pct,
                "close_pct" => r.close_pct,
                "high_from_open_pct" => r.high_from_open_pct,
                "low_from_open_pct" => r.low_from_open_pct,
                "close_from_open_pct" => r.close_from_open_pct,
                "intraday_range_pct" => r.intraday_range_pct,
                "close_position_in_range" => r.close_position_in_range,
                "sma_5" => r.sma_5,
                "sma_10" => r.sma_10,
                "sma_20" => r.sma_20,
                "macd_line" => r.macd_line,
                "macd_signal" => r.macd_signal,
                "macd_histogram" => r.macd_histogram,
                "macd_weekly_line" => r.macd_weekly_line,
                "macd_weekly_signal" => r.macd_weekly_signal,
                "macd_monthly_line" => r.macd_monthly_line,
                "macd_monthly_signal" => r.macd_monthly_signal,
                "rsi_14" => r.rsi_14,
                "kdj_k" => r.kdj_k,
                "kdj_d" => r.kdj_d,
                "kdj_j" => r.kdj_j,
                "bb_upper" => r.bb_upper,
                "bb_middle" => r.bb_middle,
                "bb_lower" => r.bb_lower,
                "bb_bandwidth" => r.bb_bandwidth,
                "bb_percent_b" => r.bb_percent_b,
                "atr" => r.atr,
                "volatility_5" => r.volatility_5,
                "volatility_20" => r.volatility_20,
                "asi" => r.asi,
                "obv" => r.obv,
                "volume_ratio" => r.volume_ratio,
                "price_momentum_5" => r.price_momentum_5,
                "price_momentum_10" => r.price_momentum_10,
                "price_momentum_20" => r.price_momentum_20,
                "price_position_52w" => r.price_position_52w,
                "body_size" => r.body_size,
                "upper_shadow" => r.upper_shadow,
                "lower_shadow" => r.lower_shadow,
                "trend_strength" => r.trend_strength,
                "adx_14" => r.adx_14,
                "vwap_distance_pct" => r.vwap_distance_pct,
                "cmf_20" => r.cmf_20,
                "aroon_up_25" => r.aroon_up_25,
                "return_lag_1" => r.return_lag_1,
                "return_lag_2" => r.return_lag_2,
                "return_lag_3" => r.return_lag_3,
                "overnight_gap" => r.overnight_gap,
                "gap_pct" => r.gap_pct,
                "volume_roc_5" => r.volume_roc_5,
                "price_roc_5" => r.price_roc_5,
                "price_roc_10" => r.price_roc_10,
                "price_roc_20" => r.price_roc_20,
                "hist_volatility_20" => r.hist_volatility_20,
                "consecutive_days" => r.consecutive_days.map(|v| v as f64),
                "index_csi300_pct_chg" => r.index_csi300_pct_chg,
                "index_csi300_vs_ma5_pct" => r.index_csi300_vs_ma5_pct,
                "index_csi300_vs_ma20_pct" => r.index_csi300_vs_ma20_pct,
                "index_chinext_pct_chg" => r.index_chinext_pct_chg,
                "index_chinext_vs_ma5_pct" => r.index_chinext_vs_ma5_pct,
                "index_chinext_vs_ma20_pct" => r.index_chinext_vs_ma20_pct,
                "index_xin9_pct_chg" => r.index_xin9_pct_chg,
                "index_xin9_vs_ma5_pct" => r.index_xin9_vs_ma5_pct,
                "index_xin9_vs_ma20_pct" => r.index_xin9_vs_ma20_pct,
                "index_hsi_pct_chg" => r.index_hsi_pct_chg,
                "index_hsi_vs_ma5_pct" => r.index_hsi_vs_ma5_pct,
                "index_hsi_vs_ma20_pct" => r.index_hsi_vs_ma20_pct,
                "fx_usdcnh_pct_chg" => r.fx_usdcnh_pct_chg,
                "fx_usdcnh_vs_ma5_pct" => r.fx_usdcnh_vs_ma5_pct,
                "fx_usdcnh_vs_ma20_pct" => r.fx_usdcnh_vs_ma20_pct,
                "net_mf_vol" => r.net_mf_vol,
                "net_mf_amount" => r.net_mf_amount,
                "smart_money_ratio" => r.smart_money_ratio,
                "large_order_flow" => r.large_order_flow,
                "industry_avg_return" => r.industry_avg_return,
                "stock_vs_industry" => r.stock_vs_industry,
                "industry_momentum_5d" => r.industry_momentum_5d,
                "industry_momentum" => r.industry_momentum,
                "turnover_rate" => r.turnover_rate,
                "turnover_rate_f" => r.turnover_rate_f,
                "pe" => r.pe,
                "pe_ttm" => r.pe_ttm,
                "pb" => r.pb,
                "dv_ratio" => r.dv_ratio,
                "dv_ttm" => r.dv_ttm,
                "total_share" => r.total_share,
                "float_share" => r.float_share,
                "free_share" => r.free_share,
                "vol_percentile" => r.vol_percentile,
                "pe_percentile_52w" => r.pe_percentile_52w,
                "sector_momentum_vs_market" => r.sector_momentum_vs_market,
                "volume_accel_5d" => r.volume_accel_5d,
                "price_vs_52w_high" => r.price_vs_52w_high,
                "consecutive_up_days" => r.consecutive_up_days.map(|v| v as f64),
                _ => None,
            };
            data[i].push(v.unwrap_or(f64::NAN));
        }
    }

    // Impute medians for NaNs
    for col in data.iter_mut() {
        let mut vals: Vec<f64> = col.iter().cloned().filter(|x| x.is_finite()).collect();
        let med = median(&mut vals);
        for x in col.iter_mut() { if !x.is_finite() { *x = med; } }
    }

    // Winsorize heavy-tailed features (clip tails) before log1p/standardize
    // Use symmetric tails at p = 0.01 (1%) for 1st and 99th percentiles
    const WINSORIZE_COLS: &[&str] = &[
        "macd_signal", "macd_weekly_signal", "macd_monthly_signal",
        "volatility_5", "volatility_20",
        "open_pct", "amount", "volume",
        "sma_5", "obv",
        "close_pct", "low_pct", "high_from_open_pct",
        "price_momentum_5",
    ];
    let mut winsor_bounds: std::collections::HashMap<&str, (f64,f64)> = std::collections::HashMap::new();
    let p = 0.01_f64;
    for (i, col) in NUMERIC_COLS.iter().enumerate() {
        if WINSORIZE_COLS.contains(col) {
            let mut vals: Vec<f64> = data[i].iter().cloned().filter(|v| v.is_finite()).collect();
            if !vals.is_empty() {
                vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
                let n = vals.len();
                let lo_idx = (p * (n as f64)).floor() as usize;
                let hi_idx = ((1.0 - p) * (n as f64)).ceil() as usize;
                let lo = vals.get(lo_idx).cloned().unwrap_or(vals[0]);
                let hi = vals.get(std::cmp::min(hi_idx, n-1)).cloned().unwrap_or(*vals.last().unwrap());
                // apply clipping
                for x in data[i].iter_mut() {
                    if *x < lo { *x = lo; }
                    if *x > hi { *x = hi; }
                }
                winsor_bounds.insert(*col, (lo, hi));
            }
        }
    }
    // persist winsor bounds for reproducibility
    std::fs::create_dir_all("artifacts")?;
    let mut wf = File::create("artifacts/winsor_params.csv")?;
    writeln!(wf, "feature,lo,hi")?;
    for (k, (lo, hi)) in winsor_bounds.iter() {
        writeln!(wf, "{},{},{}", k, lo, hi)?;
    }

    // Optionally apply log1p to selected columns
    let mut log1p_mask: Vec<bool> = vec![false; n_features];
    if apply_log1p {
        let log1p_set: std::collections::HashSet<&str> = LOG1P_COLS.iter().copied().collect();
        for (i, col) in NUMERIC_COLS.iter().enumerate() {
            if log1p_set.contains(col) {
                log1p_mask[i] = true;
                for x in data[i].iter_mut() {
                    if x.is_finite() && *x > 0.0 {
                        *x = (*x + 1.0).ln();
                    }
                }
            }
        }
    }

    // If standardize requested, compute mean/std and z-score
    let mut means: Vec<f64> = vec![0.0; n_features];
    let mut stds: Vec<f64> = vec![1.0; n_features];
    if standardize {
        for i in 0..n_features {
            let col = &data[i];
            let sum: f64 = col.iter().sum();
            let mean = sum / (col.len() as f64);
            let mut ss = 0.0f64;
            for &v in col.iter() { ss += (v - mean) * (v - mean); }
            let var = if col.len() > 0 { ss / (col.len() as f64) } else { 0.0 };
            let sd = var.sqrt();
            let sd = if sd == 0.0 { 1.0 } else { sd };
            means[i] = mean;
            stds[i] = sd;
            for x in data[i].iter_mut() {
                *x = (*x - mean) / sd;
            }
        }

        // Write scale parameters to artifacts/scale_params.csv
        std::fs::create_dir_all("artifacts")?;
        let mut f = File::create("artifacts/scale_params.csv")?;
        writeln!(f, "feature,applied_log1p,mean,std")?;
        for (i, col) in NUMERIC_COLS.iter().enumerate() {
            writeln!(f, "{},{},{},{}", col, if log1p_mask[i] {1} else {0}, means[i], stds[i])?;
        }
    }

    // Set transformed values back into FeatureRow structs
    for (row_idx, row) in rows.iter_mut().enumerate() {
        for (i, col) in NUMERIC_COLS.iter().enumerate() {
            let v = data[i][row_idx];
            match *col {
                "volume" => { row.volume = v; }
                "amount" => { row.amount = Some(v); }
                "open_pct" => { row.open_pct = Some(v); }
                "high_pct" => { row.high_pct = Some(v); }
                "low_pct" => { row.low_pct = Some(v); }
                "close_pct" => { row.close_pct = Some(v); }
                "high_from_open_pct" => { row.high_from_open_pct = Some(v); }
                "low_from_open_pct" => { row.low_from_open_pct = Some(v); }
                "close_from_open_pct" => { row.close_from_open_pct = Some(v); }
                "intraday_range_pct" => { row.intraday_range_pct = Some(v); }
                "close_position_in_range" => { row.close_position_in_range = Some(v); }
                "sma_5" => { row.sma_5 = Some(v); }
                "sma_10" => { row.sma_10 = Some(v); }
                "sma_20" => { row.sma_20 = Some(v); }
                "sma_5" => { row.sma_5 = Some(v); }
                "sma_10" => { row.sma_10 = Some(v); }
                "sma_20" => { row.sma_20 = Some(v); }
                "macd_line" => { row.macd_line = Some(v); }
                "macd_signal" => { row.macd_signal = Some(v); }
                "macd_histogram" => { row.macd_histogram = Some(v); }
                "macd_weekly_line" => { row.macd_weekly_line = Some(v); }
                "macd_weekly_signal" => { row.macd_weekly_signal = Some(v); }
                "macd_monthly_line" => { row.macd_monthly_line = Some(v); }
                "macd_monthly_signal" => { row.macd_monthly_signal = Some(v); }
                "rsi_14" => { row.rsi_14 = Some(v); }
                "kdj_k" => { row.kdj_k = Some(v); }
                "kdj_d" => { row.kdj_d = Some(v); }
                "kdj_j" => { row.kdj_j = Some(v); }
                "bb_upper" => { row.bb_upper = Some(v); }
                "bb_middle" => { row.bb_middle = Some(v); }
                "bb_lower" => { row.bb_lower = Some(v); }
                "bb_bandwidth" => { row.bb_bandwidth = Some(v); }
                "bb_percent_b" => { row.bb_percent_b = Some(v); }
                "atr" => { row.atr = Some(v); }
                "volatility_5" => { row.volatility_5 = Some(v); }
                "volatility_20" => { row.volatility_20 = Some(v); }
                "asi" => { row.asi = Some(v); }
                "obv" => { row.obv = Some(v); }
                "volume_ratio" => { row.volume_ratio = Some(v); }
                "price_momentum_5" => { row.price_momentum_5 = Some(v); }
                "price_momentum_10" => { row.price_momentum_10 = Some(v); }
                "price_momentum_20" => { row.price_momentum_20 = Some(v); }
                "price_position_52w" => { row.price_position_52w = Some(v); }
                "body_size" => { row.body_size = Some(v); }
                "upper_shadow" => { row.upper_shadow = Some(v); }
                "lower_shadow" => { row.lower_shadow = Some(v); }
                "trend_strength" => { row.trend_strength = Some(v); }
                "adx_14" => { row.adx_14 = Some(v); }
                "vwap_distance_pct" => { row.vwap_distance_pct = Some(v); }
                "cmf_20" => { row.cmf_20 = Some(v); }
                "aroon_up_25" => { row.aroon_up_25 = Some(v); }
                "return_lag_1" => { row.return_lag_1 = Some(v); }
                "return_lag_2" => { row.return_lag_2 = Some(v); }
                "return_lag_3" => { row.return_lag_3 = Some(v); }
                "overnight_gap" => { row.overnight_gap = Some(v); }
                "gap_pct" => { row.gap_pct = Some(v); }
                "volume_roc_5" => { row.volume_roc_5 = Some(v); }
                "price_roc_5" => { row.price_roc_5 = Some(v); }
                "price_roc_10" => { row.price_roc_10 = Some(v); }
                "price_roc_20" => { row.price_roc_20 = Some(v); }
                "hist_volatility_20" => { row.hist_volatility_20 = Some(v); }
                "consecutive_days" => { row.consecutive_days = Some(v.round() as i32); }
                "index_csi300_pct_chg" => { row.index_csi300_pct_chg = Some(v); }
                "index_csi300_vs_ma5_pct" => { row.index_csi300_vs_ma5_pct = Some(v); }
                "index_csi300_vs_ma20_pct" => { row.index_csi300_vs_ma20_pct = Some(v); }
                "index_chinext_pct_chg" => { row.index_chinext_pct_chg = Some(v); }
                "index_chinext_vs_ma5_pct" => { row.index_chinext_vs_ma5_pct = Some(v); }
                "index_chinext_vs_ma20_pct" => { row.index_chinext_vs_ma20_pct = Some(v); }
                "index_xin9_pct_chg" => { row.index_xin9_pct_chg = Some(v); }
                "index_xin9_vs_ma5_pct" => { row.index_xin9_vs_ma5_pct = Some(v); }
                "index_xin9_vs_ma20_pct" => { row.index_xin9_vs_ma20_pct = Some(v); }
                "index_hsi_pct_chg" => { row.index_hsi_pct_chg = Some(v); }
                "index_hsi_vs_ma5_pct" => { row.index_hsi_vs_ma5_pct = Some(v); }
                "index_hsi_vs_ma20_pct" => { row.index_hsi_vs_ma20_pct = Some(v); }
                "fx_usdcnh_pct_chg" => { row.fx_usdcnh_pct_chg = Some(v); }
                "fx_usdcnh_vs_ma5_pct" => { row.fx_usdcnh_vs_ma5_pct = Some(v); }
                "fx_usdcnh_vs_ma20_pct" => { row.fx_usdcnh_vs_ma20_pct = Some(v); }
                "net_mf_vol" => { row.net_mf_vol = Some(v); }
                "net_mf_amount" => { row.net_mf_amount = Some(v); }
                "smart_money_ratio" => { row.smart_money_ratio = Some(v); }
                "large_order_flow" => { row.large_order_flow = Some(v); }
                "industry_avg_return" => { row.industry_avg_return = Some(v); }
                "stock_vs_industry" => { row.stock_vs_industry = Some(v); }
                "industry_momentum_5d" => { row.industry_momentum_5d = Some(v); }
                "industry_momentum" => { row.industry_momentum = Some(v); }
                "turnover_rate" => { row.turnover_rate = Some(v); }
                "turnover_rate_f" => { row.turnover_rate_f = Some(v); }
                "pe" => { row.pe = Some(v); }
                "pe_ttm" => { row.pe_ttm = Some(v); }
                "pb" => { row.pb = Some(v); }
                "dv_ratio" => { row.dv_ratio = Some(v); }
                "dv_ttm" => { row.dv_ttm = Some(v); }
                "total_share" => { row.total_share = Some(v); }
                "float_share" => { row.float_share = Some(v); }
                "free_share" => { row.free_share = Some(v); }
                "vol_percentile" => { row.vol_percentile = Some(v); }
                "pe_percentile_52w" => { row.pe_percentile_52w = Some(v); }
                "sector_momentum_vs_market" => { row.sector_momentum_vs_market = Some(v); }
                "volume_accel_5d" => { row.volume_accel_5d = Some(v); }
                "price_vs_52w_high" => { row.price_vs_52w_high = Some(v); }
                "consecutive_up_days" => { row.consecutive_up_days = Some(v.round() as i32); }
                _ => {}
            }
        }
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct FeatureRow {
    ts_code: String,
    trade_date: String,
    industry: Option<String>,
    act_ent_type: Option<String>,
    volume: f64,
    amount: Option<f64>,
    month: Option<i16>,
    weekday: Option<i16>,
    quarter: Option<i16>,
    week_no: Option<i16>,
    open_pct: Option<f64>,
    high_pct: Option<f64>,
    low_pct: Option<f64>,
    close_pct: Option<f64>,
    high_from_open_pct: Option<f64>,
    low_from_open_pct: Option<f64>,
    close_from_open_pct: Option<f64>,
    intraday_range_pct: Option<f64>,
    close_position_in_range: Option<f64>,
    sma_5: Option<f64>,
    sma_10: Option<f64>,
    sma_20: Option<f64>,
    macd_line: Option<f64>,
    macd_signal: Option<f64>,
    macd_histogram: Option<f64>,
    macd_weekly_line: Option<f64>,
    macd_weekly_signal: Option<f64>,
    macd_monthly_line: Option<f64>,
    macd_monthly_signal: Option<f64>,
    rsi_14: Option<f64>,
    kdj_k: Option<f64>,
    kdj_d: Option<f64>,
    kdj_j: Option<f64>,
    bb_upper: Option<f64>,
    bb_middle: Option<f64>,
    bb_lower: Option<f64>,
    bb_bandwidth: Option<f64>,
    bb_percent_b: Option<f64>,
    atr: Option<f64>,
    volatility_5: Option<f64>,
    volatility_20: Option<f64>,
    asi: Option<f64>,
    obv: Option<f64>,
    volume_ratio: Option<f64>,
    price_momentum_5: Option<f64>,
    price_momentum_10: Option<f64>,
    price_momentum_20: Option<f64>,
    price_position_52w: Option<f64>,
    body_size: Option<f64>,
    upper_shadow: Option<f64>,
    lower_shadow: Option<f64>,
    trend_strength: Option<f64>,
    adx_14: Option<f64>,
    vwap_distance_pct: Option<f64>,
    cmf_20: Option<f64>,
    aroon_up_25: Option<f64>,
    return_lag_1: Option<f64>,
    return_lag_2: Option<f64>,
    return_lag_3: Option<f64>,
    overnight_gap: Option<f64>,
    gap_pct: Option<f64>,
    volume_roc_5: Option<f64>,
    volume_spike: Option<bool>,
    price_roc_5: Option<f64>,
    price_roc_10: Option<f64>,
    price_roc_20: Option<f64>,
    hist_volatility_20: Option<f64>,
    is_doji: Option<bool>,
    is_hammer: Option<bool>,
    is_shooting_star: Option<bool>,
    consecutive_days: Option<i32>,
    index_csi300_pct_chg: Option<f64>,
    index_csi300_vs_ma5_pct: Option<f64>,
    index_csi300_vs_ma20_pct: Option<f64>,
    index_chinext_pct_chg: Option<f64>,
    index_chinext_vs_ma5_pct: Option<f64>,
    index_chinext_vs_ma20_pct: Option<f64>,
    index_xin9_pct_chg: Option<f64>,
    index_xin9_vs_ma5_pct: Option<f64>,
    index_xin9_vs_ma20_pct: Option<f64>,
    index_hsi_pct_chg: Option<f64>,
    index_hsi_vs_ma5_pct: Option<f64>,
    index_hsi_vs_ma20_pct: Option<f64>,
    fx_usdcnh_pct_chg: Option<f64>,
    fx_usdcnh_vs_ma5_pct: Option<f64>,
    fx_usdcnh_vs_ma20_pct: Option<f64>,

    // Money flow features
    net_mf_vol: Option<f64>,
    net_mf_amount: Option<f64>,
    smart_money_ratio: Option<f64>,
    large_order_flow: Option<f64>,

    // Industry features
    industry_avg_return: Option<f64>,
    stock_vs_industry: Option<f64>,
    industry_momentum_5d: Option<f64>,
    // Industry momentum: yesterday's ChiNext pct_chg for the stock's industry
    industry_momentum: Option<f64>,

    vol_percentile: Option<f64>,
    high_vol_regime: Option<i16>,
    next_day_return: Option<f64>,
    next_day_direction: Option<i16>,
    next_3day_return: Option<f64>,
    next_3day_direction: Option<i16>,
    turnover_rate: Option<f64>,
    turnover_rate_f: Option<f64>,
    pe: Option<f64>,
    pe_ttm: Option<f64>,
    pb: Option<f64>,
    dv_ratio: Option<f64>,
    dv_ttm: Option<f64>,
    total_share: Option<f64>,
    float_share: Option<f64>,
    free_share: Option<f64>,

    // NEW: 5 predictive features for accuracy improvement
    pe_percentile_52w: Option<f64>,
    sector_momentum_vs_market: Option<f64>,
    volume_accel_5d: Option<f64>,
    price_vs_52w_high: Option<f64>,
    consecutive_up_days: Option<i32>,

    // Embeddings (deterministic hash-based, dim=8)
    industry_emb: [f64; 8],
    act_ent_type_emb: [f64; 8],

    // Imputation flags for previously-missing fields
    close_position_in_range_imputed: Option<bool>,
    macd_monthly_line_imputed: Option<bool>,
    macd_monthly_signal_imputed: Option<bool>,
}


#[derive(Clone, Debug)]
struct DailyBasic {
    ts_code: String,
    trade_date: String,
    turnover_rate: f64,
    turnover_rate_f: f64,
    volume_ratio: f64,
    pe: f64,
    pe_ttm: f64,
    pb: f64,
    dv_ratio: f64,
    dv_ttm: f64,
    total_share: f64,
    float_share: f64,
    free_share: f64,
}

// Helper struct for moneyflow data
#[derive(Clone, Debug)]
struct MoneyflowData {
    ts_code: String,
    trade_date: String,
    net_mf_vol: Option<f64>,
    net_mf_amount: Option<f64>,
}

// Helper to prefetch all moneyflow data for a stock in a date range
async fn prefetch_moneyflow_map(
    pool: &Pool<Postgres>,
    ts_code: &str,
    min_date: &str,
    max_date: &str,
) -> HashMap<(String, String), MoneyflowData> {
    let rows = sqlx::query!(
        r#"
        SELECT COALESCE(ts_code, '') as "ts_code!",
               COALESCE(trade_date, '') as "trade_date!",
            COALESCE(net_mf_vol::DOUBLE PRECISION, 0.0) as "net_mf_vol!",
            COALESCE(net_mf_amount::DOUBLE PRECISION, 0.0) as "net_mf_amount!"
        FROM moneyflow
        WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
        "#,
        ts_code,
        min_date,
        max_date
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut map = HashMap::new();
    for row in rows {
        map.insert(
            (row.ts_code.clone(), row.trade_date.clone()),
            MoneyflowData {
                ts_code: row.ts_code,
                trade_date: row.trade_date,
                net_mf_vol: Some(row.net_mf_vol),
                net_mf_amount: Some(row.net_mf_amount),
            },
        );
    }
    map
}

// Helper to prefetch all daily_basic data for a stock in a date range
async fn prefetch_daily_basic_map(
    pool: &Pool<Postgres>,
    ts_code: &str,
    min_date: &str,
    max_date: &str,
) -> HashMap<(String, String), DailyBasic> {
    let rows = sqlx::query!(
        r#"
        SELECT COALESCE(ts_code, '') as "ts_code!",
               COALESCE(trade_date, '') as "trade_date!",
            COALESCE(turnover_rate::DOUBLE PRECISION, 0.0) as "turnover_rate!",
            COALESCE(turnover_rate_f::DOUBLE PRECISION, 0.0) as "turnover_rate_f!",
            COALESCE(volume_ratio::DOUBLE PRECISION, 1.0) as "volume_ratio!",
            COALESCE(pe::DOUBLE PRECISION, 0.0) as "pe!",
            COALESCE(pe_ttm::DOUBLE PRECISION, 0.0) as "pe_ttm!",
            COALESCE(pb::DOUBLE PRECISION, 0.0) as "pb!",
            COALESCE(dv_ratio::DOUBLE PRECISION, 0.0) as "dv_ratio!",
            COALESCE(dv_ttm::DOUBLE PRECISION, 0.0) as "dv_ttm!",
            COALESCE(total_share::DOUBLE PRECISION, 0.0) as "total_share!",
            COALESCE(float_share::DOUBLE PRECISION, 0.0) as "float_share!",
            COALESCE(free_share::DOUBLE PRECISION, 0.0) as "free_share!"
        FROM daily_basic
        WHERE ts_code = $1 AND trade_date >= $2 AND trade_date <= $3
        "#,
        ts_code,
        min_date,
        max_date
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut map = HashMap::new();
    for row in rows {
        map.insert(
            (row.ts_code.clone(), row.trade_date.clone()),
            DailyBasic {
                ts_code: row.ts_code,
                trade_date: row.trade_date,
                turnover_rate: row.turnover_rate,
                turnover_rate_f: row.turnover_rate_f,
                volume_ratio: row.volume_ratio,
                pe: row.pe,
                pe_ttm: row.pe_ttm,
                pb: row.pb,
                dv_ratio: row.dv_ratio,
                dv_ttm: row.dv_ttm,
                total_share: row.total_share,
                float_share: row.float_share,
                free_share: row.free_share,
            },
        );
    }
    map
}

async fn calculate_features_for_stock_sync(
    pool: &Pool<Postgres>,
    ts_code: &str,
    list_date: Option<&str>,
    industry: Option<&str>,
    act_ent_type: Option<&str>,
    industry_perf_data: &std::sync::Arc<std::collections::HashMap<String, Vec<(String, (f64, f64))>>>,
    min_date: &str,
    max_date: &str,
    index_data: &HashMap<(String, String), IndexDaily>,
    history_years: usize,
    verbose: bool,
    leak_check: bool,
    current_date: &str,
) -> Vec<FeatureRow> {
    let _stock_start = std::time::Instant::now();

    // Collector for any leakage issues detected during feature computation
    let leak_issues = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

    // For feature calculation, we need historical context (252+ trading days for 52-week features)
    // Fetch data from min_date - 400 days to ensure we have enough history
    let fetch_start = chrono::NaiveDate::parse_from_str(min_date, "%Y%m%d")
        .ok()
        .and_then(|d| {
            let adjusted_date = (d - chrono::Duration::days(400))
                .format("%Y%m%d")
                .to_string();
            if adjusted_date.as_str() > current_date {
                println!("⚠️  fetch_start exceeds current date. Adjusting to {}.", current_date);
                Some(current_date.to_string())
            } else {
                Some(adjusted_date)
            }
        })
        .unwrap_or_else(|| "20100101".to_string());

    // Fetch all adjusted daily data for this stock (with historical buffer)
    let fetch_time = std::time::Instant::now();
    let daily_data = fetch_adjusted_daily_data(pool, ts_code, &fetch_start, max_date).await;
    let _fetch_elapsed = fetch_time.elapsed().as_millis();
    if daily_data.is_empty() {
        return vec![];
    }

    // If the stock has less than the desired history, warn but continue — we will still
    // generate features for available dates and let export-time filtering handle shorter histories.
    // Require `history_years` years worth of trading days (approx 240 trading days per year) for the warning threshold
    let min_history_required = std::cmp::max(60, (history_years * 240) as usize);
    if daily_data.len() < min_history_required {
        eprintln!(
            "⚠️  Stock {} has only {} days of history (need {} for {} years). Continuing and generating available features (will be filtered at export).",
            ts_code,
            daily_data.len(),
            min_history_required,
            history_years
        );
    }

    // --- Prefetch daily_basic and moneyflow data for this stock (with historical buffer) ---
    let daily_basic_map = prefetch_daily_basic_map(pool, ts_code, &fetch_start, max_date).await;
    let moneyflow_map = prefetch_moneyflow_map(pool, ts_code, &fetch_start, max_date).await;

    // Helper closures to access daily_basic and moneyflow with leak_check instrumentation
    let get_daily_basic_for_date = |dt: &str| {
        if leak_check {
            if dt > current_date {
                leak_issues.lock().unwrap().push(format!(
                    "daily_basic selected date {} > CURRENT_DATE {} for {}",
                    dt, current_date, ts_code
                ));
            }
        }
        daily_basic_map.get(&(ts_code.to_string(), dt.to_string()))
    };

    let get_moneyflow_for_date = |dt: &str| {
        if leak_check {
            if dt > current_date {
                leak_issues.lock().unwrap().push(format!(
                    "moneyflow selected date {} > CURRENT_DATE {} for {}",
                    dt, current_date, ts_code
                ));
            }
        }
        moneyflow_map.get(&(ts_code.to_string(), dt.to_string()))
    };

    let mut features = Vec::with_capacity(daily_data.len());
    let mut closes = Vec::new();
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    let mut opens = Vec::new();
    let mut volumes = Vec::new();
    let mut amounts = Vec::new();
    let mut pct_chgs = Vec::new();

    // For volatility percentile and regime
    let mut rolling_vols: Vec<f64> = Vec::with_capacity(daily_data.len());

    for (i, day) in daily_data.iter().enumerate() {
        let close = day.close;
        let high = day.high;
        let low = day.low;
        let open = day.open;
        let volume = day.volume;
        let amount = day.amount.unwrap_or(0.0);
        let pct_chg = day.pct_chg.unwrap_or(0.0);

        closes.push(close);
        highs.push(high);
        lows.push(low);
        opens.push(open);
        volumes.push(volume);
        amounts.push(amount);
        pct_chgs.push(pct_chg);

        // Time features
        let (month, weekday, quarter, week_no) = extract_time_features(&day.trade_date);

        // Price features (relative to previous close)
        let pre_close = if i > 0 { closes[i - 1] } else { day.open };
        let open_pct = Some((day.open - pre_close) / pre_close * 100.0);
        let high_pct = Some((day.high - pre_close) / pre_close * 100.0);
        let low_pct = Some((day.low - pre_close) / pre_close * 100.0);
        let close_pct = Some((day.close - pre_close) / pre_close * 100.0);

        // Intraday features
        let high_from_open_pct = Some((day.high - day.open) / day.open * 100.0);
        let low_from_open_pct = Some((day.low - day.open) / day.open * 100.0);
        let close_from_open_pct = Some((day.close - day.open) / day.open * 100.0);
        let intraday_range_pct = Some((day.high - day.low) / day.open * 100.0);
        let close_position_in_range = if (day.high - day.low).abs() > 1e-6 {
            Some((day.close - day.low) / (day.high - day.low) * 100.0)
        } else {
            None
        };

        // Moving averages (EMAs removed from dataset; using SMAs instead)
        // --- Add SMA features using ta::indicators::SimpleMovingAverage ---
        let sma_5 = if closes.len() >= 5 {
            let mut sma = Sma::new(5).unwrap();
            for c in &closes[(closes.len() - 5)..(closes.len() - 1)] {
                sma.next(*c);
            }
            Some(sma.next(close))
        } else {
            None
        };
        let sma_10 = if closes.len() >= 10 {
            let mut sma = Sma::new(10).unwrap();
            for c in &closes[(closes.len() - 10)..(closes.len() - 1)] {
                sma.next(*c);
            }
            Some(sma.next(close))
        } else {
            None
        };
        let sma_20 = if closes.len() >= 20 {
            let mut sma = Sma::new(20).unwrap();
            for c in &closes[(closes.len() - 20)..(closes.len() - 1)] {
                sma.next(*c);
            }
            Some(sma.next(close))
        } else {
            None
        };

        // MACD (daily)
        let (macd_line, macd_signal) = if closes.len() >= 26 {
            let (line, signal) = calculate_macd_custom(&closes, 12, 26);
            (line, signal)
        } else {
            (None, None)
        };
        let macd_histogram = match (macd_line, macd_signal) {
            (Some(line), Some(signal)) => Some(line - signal),
            _ => None,
        };

        // MACD (weekly, monthly)
        let (macd_weekly_line, macd_weekly_signal) = if closes.len() >= 130 {
            calculate_macd_custom(&closes, 60, 130) // ~12 weeks / 26 weeks
        } else {
            (None, None)
        };
        let (macd_monthly_line, macd_monthly_signal) = if closes.len() >= 260 {
            calculate_macd_custom(&closes, 126, 260) // ~6 months / 12 months
        } else {
            (None, None)
        };

        // RSI
        let rsi_14 = if closes.len() >= 14 {
            let mut rsi = Rsi::new(14).unwrap();
            for c in &closes[(closes.len() - 14)..(closes.len() - 1)] {
                rsi.next(*c);
            }
            Some(rsi.next(day.close))
        } else {
            None
        };

        // KDJ
        let (kdj_k, kdj_d, kdj_j) = if closes.len() >= 9 {
            let mut kdj = KDJIndicator::new(9).unwrap();
            for j in (closes.len() - 9)..(closes.len() - 1) {
                let di = DataItem::builder()
                    .high(highs[j])
                    .low(lows[j])
                    .close(closes[j])
                    .open(opens[j])
                    .volume(volumes[j])
                    .build()
                    .unwrap();
                kdj.next(&di);
            }
            let di = DataItem::builder()
                .high(day.high)
                .low(day.low)
                .close(day.close)
                .open(day.open)
                .volume(day.volume)
                .build()
                .unwrap();
            let kdj_vals = kdj.next(&di);
            (Some(kdj_vals.k), Some(kdj_vals.d), Some(kdj_vals.j))
        } else {
            (None, None, None)
        };

        // Bollinger Bands
        let (bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b) = if closes.len() >= 20 {
            let mut bb = BollingerBands::new(20, 2.0).unwrap();
            for j in (closes.len() - 20)..(closes.len() - 1) {
                bb.next(closes[j]);
            }
            let bands = bb.next(day.close);
            (
                Some(bands.upper),
                Some(bands.middle),
                Some(bands.lower),
                Some(bands.bandwidth),
                Some(bands.percent_b),
            )
        } else {
            (None, None, None, None, None)
        };

        // ATR
        let atr = if closes.len() >= 14 {
            let mut atr = Atr::new(14).unwrap();
            for j in (closes.len() - 14)..(closes.len() - 1) {
                let di = DataItem::builder()
                    .high(highs[j])
                    .low(lows[j])
                    .close(closes[j])
                    .open(opens[j])
                    .volume(volumes[j])
                    .build()
                    .unwrap();
                atr.next(&di);
            }
            Some(
                atr.next(
                    &DataItem::builder()
                        .high(day.high)
                        .low(day.low)
                        .close(day.close)
                        .open(day.open)
                        .volume(day.volume)
                        .build()
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        // ASI, OBV
        let asi = if closes.len() >= 2 {
            Some(calculate_asi_adjusted(
                &daily_data[..=i].iter().collect::<Vec<_>>(),
                0.1 * day.close,
            ))
        } else {
            None
        };
        let obv = if closes.len() >= 2 {
            Some(calculate_obv_adjusted(
                &daily_data[..=i].iter().collect::<Vec<_>>(),
            ))
        } else {
            None
        };

        // Volume ratio
        let volume_ratio = if volumes.len() >= 5 {
            let avg = volumes[volumes.len() - 5..].iter().sum::<f64>() / 5.0;
            if avg > 0.0 {
                Some(day.volume / avg)
            } else {
                None
            }
        } else {
            None
        };
        // Price momentum
        let price_momentum_5 = if closes.len() >= 6 {
            Some(day.close / closes[closes.len() - 6] - 1.0)
        } else {
            None
        };
        let price_momentum_10 = if closes.len() >= 11 {
            Some(day.close / closes[closes.len() - 11] - 1.0)
        } else {
            None
        };
        let price_momentum_20 = if closes.len() >= 21 {
            Some(day.close / closes[closes.len() - 21] - 1.0)
        } else {
            None
        };

        // Lagged returns
        let return_lag_1 = if closes.len() >= 2 {
            Some(day.close / closes[closes.len() - 2] - 1.0)
        } else {
            None
        };
        let return_lag_2 = if closes.len() >= 3 {
            Some(day.close / closes[closes.len() - 3] - 1.0)
        } else {
            None
        };
        let return_lag_3 = if closes.len() >= 4 {
            Some(day.close / closes[closes.len() - 4] - 1.0)
        } else {
            None
        };

        // Gap analysis
        let overnight_gap = if i > 0 {
            Some((day.open - closes[i - 1]) / closes[i - 1] * 100.0)
        } else {
            None
        };
        let gap_pct = if i > 0 {
            Some((day.open - closes[i - 1]) / closes[i - 1] * 100.0)
        } else {
            None
        };

        // Volume features
        let volume_roc_5 = if volumes.len() >= 6 {
            let prev = volumes[volumes.len() - 6];
            if prev.abs() > 1e-6 {
                Some(day.volume / prev - 1.0)
            } else {
                None
            }
        } else {
            None
        };
        let volume_spike = if volumes.len() >= 6 {
            let avg = volumes[volumes.len() - 6..volumes.len() - 1]
                .iter()
                .sum::<f64>()
                / 5.0;
            Some(day.volume > 2.0 * avg)
        } else {
            None
        };

        // Price ROC
        let price_roc_5 = if closes.len() >= 6 {
            Some(day.close / closes[closes.len() - 6] - 1.0)
        } else {
            None
        };
        let price_roc_10 = if closes.len() >= 11 {
            Some(day.close / closes[closes.len() - 11] - 1.0)
        } else {
            None
        };
        let price_roc_20 = if closes.len() >= 21 {
            Some(day.close / closes[closes.len() - 21] - 1.0)
        } else {
            None
        };

        // Historical volatility
        let hist_volatility_20 = if closes.len() >= 20 {
            Some(calculate_std_dev(&closes[closes.len() - 20..]))
        } else {
            None
        };

        // --- Add these calculations before pushing FeatureRow ---

        // Volatility
        let volatility_5 = if closes.len() >= 5 {
            Some(calculate_std_dev(&closes[closes.len() - 5..]))
        } else {
            None
        };
        let volatility_20 = if closes.len() >= 20 {
            Some(calculate_std_dev(&closes[closes.len() - 20..]))
        } else {
            None
        };

        // Price position 52w - Calculate position relative to 52-week high/low range
        let price_position_52w = if highs.len() >= 252 && lows.len() >= 252 {
            let week_52_high = highs[highs.len() - 252..]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let week_52_low = lows[lows.len() - 252..]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            if (week_52_high - week_52_low).abs() > 1e-6 {
                Some((day.close - week_52_low) / (week_52_high - week_52_low))
            } else {
                None
            }
        } else {
            None
        };

        // Candlestick patterns
        let body_size = Some((day.close - day.open).abs());
        let upper_shadow = Some(day.high - day.close.max(day.open));
        let lower_shadow = Some(day.open.min(day.close) - day.low);

        // Trend strength (ADX)
        let adx_14 = if highs.len() >= 15 {
            Some(calculate_adx(&highs, &lows, &closes, 14))
        } else {
            None
        };

        // VWAP distance
        let vwap_distance_pct = if highs.len() >= 20 {
            let highs_opt: Vec<Option<f64>> = highs.iter().map(|&v| Some(v)).collect();
            let lows_opt: Vec<Option<f64>> = lows.iter().map(|&v| Some(v)).collect();
            let closes_opt: Vec<Option<f64>> = closes.iter().map(|&v| Some(v)).collect();
            let volumes_opt: Vec<Option<f64>> = volumes.iter().map(|&v| Some(v)).collect();
            let vwap = calculate_vwap(&highs_opt, &lows_opt, &closes_opt, &volumes_opt, 20);
            if day.close.abs() > 1e-6 {
                Some((day.close - vwap) / day.close * 100.0)
            } else {
                None
            }
        } else {
            None
        };


        // Aroon (only up retained)
        let aroon_up_25 = if highs.len() >= 25 {
            let (up, _down) = calculate_aroon(&highs, &lows, 25);
            Some(up)
        } else {
            None
        };

        // Candlestick patterns (fix: call the functions and wrap in Some())
        let is_doji = Some(is_doji(
            Some(day.open),
            Some(day.close),
            Some(day.high),
            Some(day.low),
        ));
        let is_hammer = Some(is_hammer(
            Some(day.open),
            Some(day.close),
            Some(day.high),
            Some(day.low),
        ));
        let is_shooting_star = Some(is_shooting_star(
            Some(day.open),
            Some(day.close),
            Some(day.high),
            Some(day.low),
        ));

        // Consecutive days
        let consecutive_days = if closes.len() >= 2 {
            Some(count_consecutive_days(&closes))
        } else {
            None
        };

        // --- Index features: fetch from index_data ---
        let date = &day.trade_date;

        // Helper closure to fetch index feature by code and date with forward-fill
        // If exact date doesn't exist, use most recent available data before this date
        let get_index = |code: &str| {
            // Try exact date first
            if let Some(data) = index_data.get(&(code.to_string(), date.clone())) {
                if leak_check {
                    // exact match uses the requested date
                    let selected_date = date.clone();
                    if selected_date.as_str() > current_date {
                        leak_issues.lock().unwrap().push(format!("index {} exact-match selected date {} > CURRENT_DATE {} for {}", code, selected_date, current_date, ts_code));
                    }
                }
                return Some(data.clone());
            }

            // Forward-fill: find most recent date before current date
            let mut candidates: Vec<_> = index_data
                .iter()
                .filter(|((idx_code, idx_date), _)| idx_code == code && idx_date.as_str() < date.as_str())
                .collect();

            candidates.sort_by(|(a, _), (b, _)| b.1.cmp(&a.1)); // Sort by date descending

            if let Some(((_, sel_date), data)) = candidates.first() {
                if leak_check {
                    // sel_date should be strictly less than the target date
                    if sel_date.as_str() >= date.as_str() {
                        leak_issues.lock().unwrap().push(format!("index {} forward-fill selected date {} >= target date {} for {}", code, sel_date, date, ts_code));
                    }
                    if sel_date.as_str() > current_date {
                        leak_issues.lock().unwrap().push(format!("index {} forward-fill selected date {} > CURRENT_DATE {} for {}", code, sel_date, current_date, ts_code));
                    }
                }
                return Some((*data).clone());
            }
            None
        };

        // Helper: fetch index data for an explicit date (with forward-fill)
        let get_index_for_date = |code: &str, dt: &str| {
            if let Some(data) = index_data.get(&(code.to_string(), dt.to_string())) {
                if leak_check {
                    let sel_date = dt.to_string();
                    if sel_date.as_str() > current_date {
                        leak_issues.lock().unwrap().push(format!("index {} exact-match selected date {} > CURRENT_DATE {} for {}", code, sel_date, current_date, ts_code));
                    }
                }
                return Some(data.clone());
            }
            let mut candidates: Vec<_> = index_data
                .iter()
                .filter(|((idx_code, idx_date), _)| idx_code == code && idx_date.as_str() < dt)
                .collect();
            candidates.sort_by(|(a, _), (b, _)| b.1.cmp(&a.1)); // Sort by date descending
            if let Some(((_, sel_date), data)) = candidates.first() {
                if leak_check {
                    if sel_date.as_str() >= dt {
                        leak_issues.lock().unwrap().push(format!("index {} forward-fill selected date {} >= requested date {} for {}", code, sel_date, dt, ts_code));
                    }
                    if sel_date.as_str() > current_date {
                        leak_issues.lock().unwrap().push(format!("index {} forward-fill selected date {} > CURRENT_DATE {} for {}", code, sel_date, current_date, ts_code));
                    }
                }
                return Some((*data).clone());
            }
            None
        };

        // Helper: fetch industry performance for an industry and a target date.
        // Uses per-industry sorted vectors and binary search for O(log n) lookups.
        let get_industry_for_date = |ind: &str, target_date: &str| -> Option<(f64, f64)> {
            if let Some(vec) = industry_perf_data.get(ind) {
                // Binary search for position of target_date; we need the most recent date < target_date
                match vec.binary_search_by(|(d, _)| d.as_str().cmp(target_date)) {
                    Ok(idx) => {
                        if idx == 0 {
                            None
                        } else {
                            let selected_date = vec[idx - 1].0.clone();
                            if leak_check {
                                if selected_date.as_str() >= target_date {
                                    leak_issues.lock().unwrap().push(format!("industry {} lookup selected date {} >= target_date {} for {}", ind, selected_date, target_date, ts_code));
                                }
                                if selected_date.as_str() > current_date {
                                    leak_issues.lock().unwrap().push(format!("industry {} lookup selected date {} > CURRENT_DATE {} for {}", ind, selected_date, current_date, ts_code));
                                }
                            }
                            let vals = &vec[idx - 1].1;
                            Some(*vals)
                        }
                    }
                    Err(idx) => {
                        if idx == 0 {
                            None
                        } else {
                            let selected_date = vec[idx - 1].0.clone();
                            if leak_check {
                                if selected_date.as_str() >= target_date {
                                    leak_issues.lock().unwrap().push(format!("industry {} lookup selected date {} >= target_date {} for {}", ind, selected_date, target_date, ts_code));
                                }
                                if selected_date.as_str() > current_date {
                                    leak_issues.lock().unwrap().push(format!("industry {} lookup selected date {} > CURRENT_DATE {} for {}", ind, selected_date, current_date, ts_code));
                                }
                            }
                            let vals = &vec[idx - 1].1;
                            Some(*vals)
                        }
                    }
                }
            } else {
                None
            }
        };

        // Determine previous trading date (if available) and use it for index lookups
        let prev_date_opt = if i > 0 {
            Some(daily_data[i - 1].trade_date.clone())
        } else {
            None
        };

        // CSI300
        let csi300 = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("000300.SH", pd)
        } else {
            None
        };

        // Determine previous trading date (if available) and use it for index lookups
        let prev_date_opt = if i > 0 {
            Some(daily_data[i - 1].trade_date.clone())
        } else {
            None
        };

        // CSI300
        let csi300 = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("000300.SH", pd)
        } else {
            None
        };
        let index_csi300_pct_chg = csi300.as_ref().map(|x| Some(x.pct_chg)).flatten();
        let index_csi300_vs_ma5_pct = csi300.as_ref().and_then(|x| match x.ma5 {
            Some(ma5) if ma5.abs() > 1e-6 => Some((x.close - ma5) / ma5 * 100.0),
            _ => None,
        });
        let index_csi300_vs_ma20_pct = csi300.as_ref().and_then(|x| match x.ma20 {
            Some(ma20) if ma20.abs() > 1e-6 => Some((x.close - ma20) / ma20 * 100.0),
            _ => None,
        });

        // ChiNext
        let chinext = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("399006.SZ", pd)
        } else {
            None
        };
        let index_chinext_pct_chg = chinext.as_ref().map(|x| Some(x.pct_chg)).flatten();
        let index_chinext_vs_ma5_pct = chinext.as_ref().and_then(|x| match x.ma5 {
            Some(ma5) if ma5.abs() > 1e-6 => Some((x.close - ma5) / ma5 * 100.0),
            _ => None,
        });
        let index_chinext_vs_ma20_pct = chinext.as_ref().and_then(|x| match x.ma20 {
            Some(ma20) if ma20.abs() > 1e-6 => Some((x.close - ma20) / ma20 * 100.0),
            _ => None,
        });

        // XIN9
        let xin9 = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("XIN9", pd)
        } else {
            None
        };
        let index_xin9_pct_chg = xin9.as_ref().map(|x| Some(x.pct_chg)).flatten();
        let index_xin9_vs_ma5_pct = xin9.as_ref().and_then(|x| match x.ma5 {
            Some(ma5) if ma5.abs() > 1e-6 => Some((x.close - ma5) / ma5 * 100.0),
            _ => None,
        });
        let index_xin9_vs_ma20_pct = xin9.as_ref().and_then(|x| match x.ma20 {
            Some(ma20) if ma20.abs() > 1e-6 => Some((x.close - ma20) / ma20 * 100.0),
            _ => None,
        });

        // HSI (Hong Kong Hang Seng Index)
        let hsi = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("HSI", pd)
        } else {
            None
        };
        let index_hsi_pct_chg = hsi.as_ref().map(|x| Some(x.pct_chg)).flatten();
        let index_hsi_vs_ma5_pct = hsi.as_ref().and_then(|x| match x.ma5 {
            Some(ma5) if ma5.abs() > 1e-6 => Some((x.close - ma5) / ma5 * 100.0),
            _ => None,
        });
        let index_hsi_vs_ma20_pct = hsi.as_ref().and_then(|x| match x.ma20 {
            Some(ma20) if ma20.abs() > 1e-6 => Some((x.close - ma20) / ma20 * 100.0),
            _ => None,
        });

        // USDCNH (USD/CNH FX rate)
        let usdcnh = if let Some(ref pd) = prev_date_opt {
            get_index_for_date("USDCNH.FXCM", pd)
        } else {
            None
        };
        let fx_usdcnh_pct_chg = usdcnh.as_ref().map(|x| Some(x.pct_chg)).flatten();
        let fx_usdcnh_vs_ma5_pct = usdcnh.as_ref().and_then(|x| match x.ma5 {
            Some(ma5) if ma5.abs() > 1e-6 => Some((x.close - ma5) / ma5 * 100.0),
            _ => None,
        });
        let fx_usdcnh_vs_ma20_pct = usdcnh.as_ref().and_then(|x| match x.ma20 {
            Some(ma20) if ma20.abs() > 1e-6 => Some((x.close - ma20) / ma20 * 100.0),
            _ => None,
        });

        // industry_momentum: yesterday's ChiNext pct_chg (shift by 1 within each industry)
        let industry_momentum = if i > 0 {
            let prev_date = &daily_data[i - 1].trade_date;
            get_index_for_date("399006.SZ", prev_date)
                .as_ref()
                .map(|x| Some(x.pct_chg))
                .flatten()
        } else {
            None
        };

        // --- Volatility percentile and regime ---
        let vol_60 = if closes.len() >= 60 {
            Some(calculate_std_dev(&closes[closes.len() - 60..]))
        } else {
            None
        };
        if let Some(v) = vol_60 {
            rolling_vols.push(v);
        } else {
            rolling_vols.push(0.0);
        }
        let (vol_percentile, high_vol_regime) = if i >= 59 {
            let this_vol = rolling_vols[i];
            let mut window: Vec<f64> = rolling_vols[i.saturating_sub(59)..=i].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos = window.iter().position(|&v| v >= this_vol).unwrap_or(0);
            let percentile = (pos as f64) / (window.len() as f64);
            let regime = if percentile > 0.8 { Some(1) } else { Some(0) };
            (Some(percentile), regime)
        } else {
            (None, None)
        };

        // --- Targets: next day/3day return & direction ---
        let next_day_return = if i + 1 < daily_data.len() {
            let next_close = daily_data[i + 1].close;
            let this_close = day.close;
            if this_close.abs() > 1e-8 {
                Some((next_close / this_close) - 1.0)
            } else {
                None
            }
        } else {
            None
        };
        let next_day_direction = next_day_return.map(|r| {
            // Binary classification: 1 if up, -1 if down (no neutral class)
            if r >= 0.0 {
                1
            } else {
                -1
            }
        });

        let next_3day_return = if i + 3 < daily_data.len() {
            let next_close = daily_data[i + 3].close;
            let this_close = day.close;
            if this_close.abs() > 1e-8 {
                Some((next_close / this_close) - 1.0)
            } else {
                None
            }
        } else {
            None
        };
        let next_3day_direction = next_3day_return.map(|r| {
            // Binary classification: 1 if up, -1 if down (no neutral class)
            if r >= 0.0 {
                1
            } else {
                -1
            }
        });

        // --- Map daily_basic fields by (ts_code, trade_date) ---
        let daily_basic = get_daily_basic_for_date(&day.trade_date);

        // CMF (already computed earlier via function) - ensure we set it here
        let cmf_20 = if highs.len() >= 20 {
            Some(calculate_cmf(
                &highs.iter().map(|&v| Some(v)).collect::<Vec<Option<f64>>>(),
                &lows.iter().map(|&v| Some(v)).collect::<Vec<Option<f64>>>(),
                &closes.iter().map(|&v| Some(v)).collect::<Vec<Option<f64>>>(),
                &volumes.iter().map(|&v| Some(v)).collect::<Vec<Option<f64>>>(),
                20,
            ))
        } else {
            None
        };

        // MFI removed (insufficient historical data for reliable calculation)



        // Industry features (lookup from pre-fetched map)
        // Use most-recent industry performance strictly prior to the current `trade_date` to avoid look-ahead.
        let (industry_avg_return, industry_momentum_5d) = if let Some(ind) = industry {
            // Find latest industry perf entry with date < current trade_date (handles suspensions)
            let res = get_industry_for_date(ind, &day.trade_date);
            if let Some((avg, mom5)) = res {
                // Debug for suspicious date range to ensure strict prior-date selection
                if verbose && day.trade_date.as_str() >= "20210110" && day.trade_date.as_str() <= "20210131" {
                    println!("[DEBUG] industry lookup: ts_code={}, trade_date={}, industry={}, chosen_avg={}, chosen_mom5={}", ts_code, day.trade_date, ind, avg, mom5);
                }
                (Some(avg), Some(mom5))
            } else {
                // Debug missing prior entry
                if verbose && day.trade_date.as_str() >= "20210110" && day.trade_date.as_str() <= "20210131" {
                    println!("[DEBUG] industry lookup: ts_code={}, trade_date={}, industry={}, NO_PRIOR_FOUND", ts_code, day.trade_date, ind);
                }
                (None, None)
            }
        } else {
            (None, None)
        };

        let stock_vs_industry = match (industry_avg_return, close_pct) {
            (Some(ind_avg), Some(cp)) => Some(cp - ind_avg),
            _ => None,
        };

        // ========== NEW: 5 Predictive Features for Accuracy Improvement ==========

        // Feature 1: PE Percentile (52-week)
        // Calculates where current PE TTM sits within 52-week range (0-1 scale)
        // Defaults to 0.0 for loss-making companies (negative PE) to avoid NULL values
        let pe_percentile_52w = if i >= 251
            && daily_basic
                .as_ref()
                .map(|db| db.pe_ttm > 0.0)
                .unwrap_or(false)
        {
            let pe_ttm = daily_basic.as_ref().unwrap().pe_ttm;
            let mut pe_values: Vec<f64> = Vec::new();

            // Collect 52-week PE values (252 trading days)
            for j in (i.saturating_sub(251))..=i {
                if let Some(db) =
                    daily_basic_map.get(&(ts_code.to_string(), daily_data[j].trade_date.clone()))
                {
                    if db.pe_ttm > 0.0 {
                        // Only include positive PE values
                        pe_values.push(db.pe_ttm);
                    }
                }
            }

            if pe_values.len() > 10 {
                // Need reasonable sample size
                let pe_min = pe_values.iter().cloned().fold(f64::INFINITY, f64::min);
                let pe_max = pe_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                if pe_max > pe_min && pe_ttm > 0.0 {
                    Some((pe_ttm - pe_min) / (pe_max - pe_min))
                } else {
                    Some(0.0) // 0.0 for negative PE (loss-making)
                }
            } else {
                Some(0.0) // 0.0 if insufficient PE history
            }
        } else {
            Some(0.0) // 0.0 for loss-making companies (no positive PE)
        };

        // Feature 2: Sector Momentum vs Market
        // Compare this stock's sector performance to overall market
        let sector_momentum_vs_market = if industry.is_some() && price_momentum_5.is_some() {
            // This is a placeholder - full implementation requires sector aggregation
            // For now, we'll compute it as 0.0 and update in a second pass
            Some(0.0)
        } else {
            None
        };

        // Feature 3: Volume Acceleration (5-day)
        // Measures rate of change in volume activity
        let volume_accel_5d = if i >= 10 {
            let current_vol_ratio = volume_ratio.unwrap_or(1.0);

            // Calculate average volume_ratio from days i-10 to i-5
            let mut vol_ratios: Vec<f64> = Vec::new();
            for j in (i.saturating_sub(10))..=(i.saturating_sub(5)) {
                if volumes.len() > j && volumes[j] > 0.0 {
                    let avg_vol: f64 = volumes[j.saturating_sub(4)..=j].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 {
                        vol_ratios.push(volumes[j] / avg_vol);
                    }
                }
            }

            if !vol_ratios.is_empty() {
                let avg_past_vol_ratio = vol_ratios.iter().sum::<f64>() / vol_ratios.len() as f64;
                if avg_past_vol_ratio > 0.0 {
                    Some((current_vol_ratio - avg_past_vol_ratio) / avg_past_vol_ratio)
                } else {
                    Some(0.0)
                }
            } else {
                None
            }
        } else {
            None
        };

        // Feature 4: Price vs 52-Week High
        // Shows how far current price is from yearly high (negative = below high)
        let price_vs_52w_high = if closes.len() >= 252 {
            let high_52w = highs[highs.len() - 252..]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let current_close = day.close;

            if high_52w > 0.0 && current_close > 0.0 {
                Some((current_close - high_52w) / high_52w)
            } else {
                None
            }
        } else {
            None
        };

        // Feature 5: Consecutive Up Days
        // Count sequential up/down days (positive = up streak, negative = down streak)
        let consecutive_up_days = if closes.len() >= 2 {
            let mut count = 0;
            let is_up = closes[closes.len() - 1] > closes[closes.len() - 2];

            // Count backwards from current day
            for j in (1..closes.len().min(20)).rev() {
                // Limit to 20 days
                let current_up = closes[j] > closes[j - 1];
                if current_up == is_up {
                    count += 1;
                } else {
                    break;
                }
            }

            if is_up { Some(count) } else { Some(-count) }
        } else {
            None
        };

        // ========== END: New Features ==========

        // --- Moneyflow Features ---
        let moneyflow = get_moneyflow_for_date(&day.trade_date);

        // Raw moneyflow values
        let net_mf_vol = moneyflow.and_then(|m| m.net_mf_vol);
        let net_mf_amount = moneyflow.and_then(|m| m.net_mf_amount);

        // Compute smart_money_ratio: normalized ratio of net_mf_amount to volume
        // Measures the strength of money flow relative to trading volume
        let smart_money_ratio =
            if let (Some(mf_amount), Some(vol)) = (net_mf_amount, Some(day.volume)) {
                if vol > 0.0 {
                    Some((mf_amount / vol).clamp(-1.0, 1.0)) // Clamp to avoid extreme outliers
                } else {
                    None
                }
            } else {
                None
            };

        // Compute large_order_flow: 5-day rolling average of net moneyflow volume
        // Indicates sustained buy/sell pressure from large orders
        let large_order_flow = if i >= 4 {
            let mf_window: Vec<f64> = (i.saturating_sub(4)..=i)
                .filter_map(|j| {
                    get_moneyflow_for_date(&daily_data[j].trade_date)
                })
                .filter_map(|m| m.net_mf_vol)
                .collect();

            if !mf_window.is_empty() {
                Some(mf_window.iter().sum::<f64>() / mf_window.len() as f64)
            } else {
                None
            }
        } else {
            net_mf_vol // Use current day's value if not enough history
        };

        // ========== END: Moneyflow Features ==========

        // Only create FeatureRow for dates >= min_date (for incremental updates)
        // AND only if we have sufficient historical data for all technical indicators
        // Requirement: Need at least 20 days for Bollinger Bands, 252 for 52-week features
        // So we skip until we have 60+ days of history
        if day.trade_date.as_str() >= min_date && closes.len() >= 60 {
            features.push(FeatureRow {
                ts_code: ts_code.to_string(),
                trade_date: day.trade_date.clone(),
                industry: industry.map(|s| s.to_string()),
                act_ent_type: Some(act_ent_type.unwrap_or("UNKNOWN").to_string()),
                volume: day.volume,
                amount: day.amount,
                month,
                weekday,
                quarter,
                week_no,
                open_pct,
                high_pct,
                low_pct,
                close_pct,
                high_from_open_pct,
                low_from_open_pct,
                close_from_open_pct,
                intraday_range_pct,
                close_position_in_range,
                sma_5,
                sma_10,
                sma_20,
                macd_line,
                macd_signal,
                macd_histogram,
                macd_weekly_line,
                macd_weekly_signal,
                macd_monthly_line,
                macd_monthly_signal,
                rsi_14,
                kdj_k,
                kdj_d,
                kdj_j,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_bandwidth,
                bb_percent_b,
                atr,
                volatility_5,
                volatility_20,
                asi,
                obv,
                volume_ratio,
                price_momentum_5,
                price_momentum_10,
                price_momentum_20,
                price_position_52w,
                body_size,
                upper_shadow,
                lower_shadow,
                trend_strength: adx_14,
                adx_14,
                vwap_distance_pct,
                cmf_20,
                aroon_up_25,
                return_lag_1,
                return_lag_2,
                return_lag_3,
                overnight_gap,
                gap_pct,
                volume_roc_5,
                volume_spike,
                price_roc_5,
                price_roc_10,
                price_roc_20,
                hist_volatility_20,
                is_doji,
                is_hammer,
                is_shooting_star,
                consecutive_days,
                index_csi300_pct_chg,
                index_csi300_vs_ma5_pct,
                index_csi300_vs_ma20_pct,
                index_chinext_pct_chg,
                index_chinext_vs_ma5_pct,
                index_chinext_vs_ma20_pct,
                index_xin9_pct_chg,
                index_xin9_vs_ma5_pct,
                index_xin9_vs_ma20_pct,
                index_hsi_pct_chg,
                index_hsi_vs_ma5_pct,
                index_hsi_vs_ma20_pct,
                fx_usdcnh_pct_chg,
                fx_usdcnh_vs_ma5_pct,
                fx_usdcnh_vs_ma20_pct,

                // --- Moneyflow features ---
                net_mf_vol,
                net_mf_amount,
                smart_money_ratio,
                large_order_flow,

                // --- Industry features ---
                industry_avg_return,
                stock_vs_industry,
                industry_momentum_5d,
                industry_momentum,

                // --- Map daily_basic fields by date ---
                turnover_rate: daily_basic.as_ref().map(|db| db.turnover_rate),
                turnover_rate_f: daily_basic.as_ref().map(|db| db.turnover_rate_f),
                pe: daily_basic.as_ref().map(|db| db.pe),
                pe_ttm: daily_basic.as_ref().map(|db| db.pe_ttm),
                pb: daily_basic.as_ref().map(|db| db.pb),
                dv_ratio: daily_basic.as_ref().map(|db| db.dv_ratio),
                dv_ttm: Some(daily_basic.as_ref().map(|db| db.dv_ttm).unwrap_or(0.0)), // Default to 0.0 if no dividends
                total_share: daily_basic.as_ref().map(|db| db.total_share),
                float_share: daily_basic.as_ref().map(|db| db.float_share),
                free_share: daily_basic.as_ref().map(|db| db.free_share),
                // --- Add missing fields ---
                vol_percentile,
                high_vol_regime,
                next_day_return,
                next_day_direction,
                next_3day_return,
                next_3day_direction,

                // NEW: 5 predictive features
                pe_percentile_52w,
                sector_momentum_vs_market,
                volume_accel_5d,
                price_vs_52w_high,
                consecutive_up_days,

                // Embedding placeholders (filled later)
                industry_emb: [0.0; 8],
                act_ent_type_emb: [0.0; 8],

                // Imputation flags (computed post-hoc)
                close_position_in_range_imputed: None,
                macd_monthly_line_imputed: None,
                macd_monthly_signal_imputed: None,
            });
        } // end if day.trade_date >= min_date
    }
    // Runtime guard: drop any feature rows with trade_date after current_date
    let total_rows = features.len();
    let mut filtered: Vec<FeatureRow> = features
        .into_iter()
        .filter(|r| r.trade_date.as_str() <= current_date)
        .collect();
    let dropped = total_rows.saturating_sub(filtered.len());
    if dropped > 0 {
        eprintln!("⚠️  Dropped {} future-dated feature rows (CURRENT_DATE={}) for {}", dropped, current_date, ts_code);
    }

    // --- Imputation & Embedding Post-processing (per-stock) ---
    // Close position in range: fill missing with per-stock median, fallback to 0.5
    let mut close_vals: Vec<f64> = filtered.iter().filter_map(|r| r.close_position_in_range).collect();
    let close_median = if !close_vals.is_empty() {
        close_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = close_vals.len() / 2;
        if close_vals.len() % 2 == 0 {
            (close_vals[mid - 1] + close_vals[mid]) / 2.0
        } else {
            close_vals[mid]
        }
    } else {
        0.5
    };

    // MACD monthly line/signal median fallback
    let mut macd_line_vals: Vec<f64> = filtered.iter().filter_map(|r| r.macd_monthly_line).collect();
    let macd_line_median = if !macd_line_vals.is_empty() {
        macd_line_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = macd_line_vals.len() / 2;
        if macd_line_vals.len() % 2 == 0 { (macd_line_vals[mid - 1] + macd_line_vals[mid]) / 2.0 } else { macd_line_vals[mid] }
    } else { 0.0 };

    let mut macd_sig_vals: Vec<f64> = filtered.iter().filter_map(|r| r.macd_monthly_signal).collect();
    let macd_sig_median = if !macd_sig_vals.is_empty() {
        macd_sig_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = macd_sig_vals.len() / 2;
        if macd_sig_vals.len() % 2 == 0 { (macd_sig_vals[mid - 1] + macd_sig_vals[mid]) / 2.0 } else { macd_sig_vals[mid] }
    } else { 0.0 };

    // Embedding params (fixed defaults)
    let embed_dim = EMBED_DIM;
    let embed_seed = EMBED_SEED;

    for r in filtered.iter_mut() {
        // Close position imputation
        if r.close_position_in_range.is_none() {
            r.close_position_in_range = Some(close_median);
            r.close_position_in_range_imputed = Some(true);
        } else {
            r.close_position_in_range_imputed = Some(false);
        }
        // MACD monthly
        if r.macd_monthly_line.is_none() {
            r.macd_monthly_line = Some(macd_line_median);
            r.macd_monthly_line_imputed = Some(true);
        } else {
            r.macd_monthly_line_imputed = Some(false);
        }
        if r.macd_monthly_signal.is_none() {
            r.macd_monthly_signal = Some(macd_sig_median);
            r.macd_monthly_signal_imputed = Some(true);
        } else {
            r.macd_monthly_signal_imputed = Some(false);
        }

        // Embeddings (deterministic)
        let ind = r.industry.clone().unwrap_or_else(|| "UNKNOWN".to_string());
        let ind_vec = hash_to_embedding_fixed(&ind, embed_dim, embed_seed);
        for i in 0..embed_dim.min(r.industry_emb.len()) {
            r.industry_emb[i] = ind_vec.get(i).cloned().unwrap_or(0.0);
        }
        let at = r.act_ent_type.clone().unwrap_or_else(|| "UNKNOWN".to_string());
        let at_vec = hash_to_embedding_fixed(&at, embed_dim, embed_seed.wrapping_add(1));
        for i in 0..embed_dim.min(r.act_ent_type_emb.len()) {
            r.act_ent_type_emb[i] = at_vec.get(i).cloned().unwrap_or(0.0);
        }
    }

    // If leak_check was enabled and we found any lookup issues, abort with details
    if leak_check && !leak_issues.lock().unwrap().is_empty() {
        let issues = leak_issues.lock().unwrap();
        eprintln!("\n🚨 Data leakage issues detected for {}: {} issues:\n", ts_code, issues.len());
        for issue in issues.iter() {
            eprintln!(" - {}", issue);
        }
        eprintln!("Aborting due to data leakage detection (exit code 2)");
        std::process::exit(2);
    }
    filtered
}

/// Apply monthly cross-sectional winsorization to `next_day_return`.
/// This computes, for each calendar month (YYYYMM), the lower and upper percentile
/// boundaries using `percentile_cont` and clamps `next_day_return` into that range.
async fn apply_monthly_winsorization(pool: &Pool<Postgres>, pct: f64) -> Result<u64, Box<dyn Error + Send + Sync>> {
    let lower = pct;
    let upper = 1.0 - pct;
    println!("⚙️  Applying monthly winsorization: pct={} (lower={} upper={})", pct, lower, upper);

    let sql = r#"
        WITH monthly_bounds AS (
            SELECT TO_CHAR(TO_DATE(trade_date,'YYYYMMDD'),'YYYYMM') AS trade_ym,
                   percentile_cont($1) WITHIN GROUP (ORDER BY next_day_return) AS low,
                   percentile_cont($2) WITHIN GROUP (ORDER BY next_day_return) AS high
            FROM ml_training_dataset
            WHERE next_day_return IS NOT NULL
            GROUP BY trade_ym
        )
        UPDATE ml_training_dataset m
        SET next_day_return = LEAST(GREATEST(m.next_day_return, monthly_bounds.low), monthly_bounds.high)
        FROM monthly_bounds
        WHERE TO_CHAR(TO_DATE(m.trade_date,'YYYYMMDD'),'YYYYMM') = monthly_bounds.trade_ym
          AND m.next_day_return IS NOT NULL
    "#;

    let result = sqlx::query(sql).bind(lower).bind(upper).execute(pool).await?;
    let updated = result.rows_affected();
    println!("✅ Monthly winsorization complete. Rows updated: {}", updated);
    Ok(updated)
}

/// Helper function to create the machine learning training dataset table
async fn create_ml_training_dataset_table(pool: &Pool<Postgres>) -> Result<(), sqlx::Error> {
    // Add missing columns to ml_training_dataset if they do not exist
    let alter_statements = [
        // DailyBasic columns (excluding close)
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS turnover_rate DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS turnover_rate_f DOUBLE PRECISION;",
        // Do NOT add volume_ratio here, it's already present as a feature column
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS pe DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS pe_ttm DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS pb DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS dv_ratio DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS dv_ttm DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS total_share DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS float_share DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS free_share DOUBLE PRECISION;",
        // Index HSI columns
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS index_hsi_pct_chg DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS index_hsi_vs_ma5_pct DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS index_hsi_vs_ma20_pct DOUBLE PRECISION;",
        // FX USDCNH columns
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS fx_usdcnh_pct_chg DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS fx_usdcnh_vs_ma5_pct DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS fx_usdcnh_vs_ma20_pct DOUBLE PRECISION;",
        // Moneyflow columns
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS net_mf_vol DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS net_mf_amount DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS smart_money_ratio DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS large_order_flow DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_avg_return DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS stock_vs_industry DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_momentum_5d DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_momentum DOUBLE PRECISION;",
        // Embedding columns (industry, act_ent_type) - fixed dim 8
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_0 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_1 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_2 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_3 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_4 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_5 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_6 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS industry_emb_7 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_0 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_1 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_2 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_3 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_4 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_5 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_6 DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS act_ent_type_emb_7 DOUBLE PRECISION;",
        // Imputation flags
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS close_position_in_range_imputed BOOLEAN;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS macd_monthly_line_imputed BOOLEAN;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS macd_monthly_signal_imputed BOOLEAN;",
    ];

    // Run CREATE with PRIMARY KEY
    sqlx::query(get_ml_create_table_sql())
        .execute(pool)
        .await?;
    

    // Run ALTER TABLEs to add missing columns (idempotent, safe if already exist)
    for stmt in alter_statements.iter() {
        sqlx::query(stmt).execute(pool).await.ok();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hash_to_embedding_fixed;
    use super::EMBED_DIM;
    use super::EMBED_SEED;

    #[test]
    fn test_hash_to_embedding_fixed_deterministic_normalized() {
        let v1 = hash_to_embedding_fixed("TEST_INDUSTRY", EMBED_DIM, EMBED_SEED);
        let v2 = hash_to_embedding_fixed("TEST_INDUSTRY", EMBED_DIM, EMBED_SEED);
        assert_eq!(v1, v2);
        let norm: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "norm is {} not ~1", norm);
    }

    #[test]
    fn test_hash_to_embedding_fixed_seed_variation_and_dim() {
        let v1 = hash_to_embedding_fixed("TEST", EMBED_DIM, EMBED_SEED);
        let v2 = hash_to_embedding_fixed("TEST", EMBED_DIM, EMBED_SEED.wrapping_add(1));
        assert_ne!(v1, v2);
        let v_small = hash_to_embedding_fixed("TEST", 4usize, EMBED_SEED);
        assert_eq!(v_small.len(), 4);
    }

    #[test]
    fn test_create_table_contains_embedding_columns() {
        let sql = super::get_ml_create_table_sql();
        assert!(sql.contains("industry_emb_0"));
        assert!(sql.contains("act_ent_type_emb_7"));
    }
}
