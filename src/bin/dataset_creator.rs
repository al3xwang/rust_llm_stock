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

    // If leak_check requested, restrict to single stock and force sequential processing
    let concurrency = if cli.leak_check { 1usize } else { cli.concurrency };
    if cli.leak_check {
        let selected = cli
            .leak_stock
            .clone()
            .or_else(|| stocks.get(0).map(|s| s.0.clone()));
        if let Some(ts) = selected {
            stocks.retain(|(code, _, _, _)| code == &ts);
            if stocks.is_empty() {
                eprintln!("⚠️  leak-check: specified stock {} not found in stock list", ts);
                std::process::exit(3);
            }
            println!("Running leak-check for {} (concurrency forced to 1)", ts);
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
    let (final_min_date, final_max_date) = if min_date > max_date {
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
    println!("\n=== Pre-fetching Industry Performance Data ===");
    let industry_perf_data =
        std::sync::Arc::new(prefetch_industry_performance(&dbpool, &final_min_date, &final_max_date).await);


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
                    batch_insert_feature_rows(&dbpool, &feature_rows).await?;
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
            intraday_range_pct, close_position_in_range, ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
            macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
            rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
            asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
            upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20, williams_r_14, aroon_up_25,
            aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, volume_roc_5, volume_spike,
            price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, is_shooting_star, consecutive_days,
            index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, index_chinext_pct_chg, index_chinext_vs_ma5_pct,
            index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
            index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
            fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
            net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow,
            turnover_rate, turnover_rate_f, /* volume_ratio, */ pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share,
            free_share, total_mv, circ_mv,
            vol_percentile, high_vol_regime, next_day_return,
            next_day_direction, next_3day_return, next_3day_direction,
            pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days
        ) VALUES (
            -- 1-126: all columns including HSI, USDCNH, and moneyflow
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80,$81,$82,$83,$84,$85,$86,$87,$88,$89,$90,$91,$92,$93,$94,$95,$96,
            $97,$98,$99,$100,$101,$102,$103,$104,$105,$106,$107,$108,$109,$110,$111,$112,$113,$114,$115,$116,$117,$118,$119,$120,$121,$122,$123,$124,$125,$126
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
    .bind(row.ema_5)
    .bind(row.ema_10)
    .bind(row.ema_20)
    .bind(row.ema_30)
    .bind(row.ema_60)
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
    .bind(row.williams_r_14)
    .bind(row.aroon_up_25)
    .bind(row.aroon_down_25)
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
    .bind(row.ps)
    .bind(row.ps_ttm)
    .bind(row.dv_ratio)
    .bind(row.dv_ttm)
    .bind(row.total_share)
    .bind(row.float_share)
    .bind(row.free_share)
    .bind(row.total_mv)
    .bind(row.circ_mv)
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

    // Process in chunks of 50 rows for faster visibility (was 500)
    const CHUNK_SIZE: usize = 50;

    for chunk in rows.chunks(CHUNK_SIZE) {
        // Build multi-value INSERT statement dynamically
        let mut sql = String::from(
            r#"INSERT INTO ml_training_dataset (
                ts_code, trade_date, industry, act_ent_type, volume, amount, month, weekday, quarter, week_no,
                open_pct, high_pct, low_pct, close_pct, high_from_open_pct, low_from_open_pct, close_from_open_pct,
                intraday_range_pct, close_position_in_range, ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
                macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
                rsi_14, kdj_k, kdj_d, kdj_j, bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b, atr, volatility_5, volatility_20,
                asi, obv, volume_ratio, price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w, body_size,
                upper_shadow, lower_shadow, trend_strength, adx_14, vwap_distance_pct, cmf_20, williams_r_14, aroon_up_25,
                aroon_down_25, return_lag_1, return_lag_2, return_lag_3, overnight_gap, gap_pct, volume_roc_5, volume_spike,
                price_roc_5, price_roc_10, price_roc_20, hist_volatility_20, is_doji, is_hammer, is_shooting_star, consecutive_days,
                index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct, index_chinext_pct_chg, index_chinext_vs_ma5_pct,
                index_chinext_vs_ma20_pct, index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
                index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
                fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
                net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow, industry_avg_return, stock_vs_industry, industry_momentum_5d, industry_momentum,
                turnover_rate, turnover_rate_f, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share,
                free_share, total_mv, circ_mv,
                vol_percentile, high_vol_regime, next_day_return,
                next_day_direction, next_3day_return, next_3day_direction,
                pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days
            ) VALUES "#,
        );

        // Generate value placeholders for each row
        let cols_per_row = 126;
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

        // Upsert: overwrite industry-related fields to ensure corrected lagged values replace previous (possibly leaked) values
        sql.push_str(" ON CONFLICT (ts_code, trade_date) DO UPDATE SET \
            industry_avg_return = EXCLUDED.industry_avg_return, \
            stock_vs_industry = EXCLUDED.stock_vs_industry, \
            industry_momentum_5d = EXCLUDED.industry_momentum_5d, \
            industry_momentum = EXCLUDED.industry_momentum");

        // Build the query with all bindings
        let mut query = sqlx::query(&sql);

        for row in chunk {
            query = query
                .bind(&row.ts_code)
                .bind(&row.trade_date)
                .bind(&row.industry)
                .bind(&row.act_ent_type)
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
                .bind(row.ema_5)
                .bind(row.ema_10)
                .bind(row.ema_20)
                .bind(row.ema_30)
                .bind(row.ema_60)
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
                .bind(row.williams_r_14)
                .bind(row.aroon_up_25)
                .bind(row.aroon_down_25)
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
                .bind(row.ps)
                .bind(row.ps_ttm)
                .bind(row.dv_ratio)
                .bind(row.dv_ttm)
                .bind(row.total_share)
                .bind(row.float_share)
                .bind(row.free_share)
                .bind(row.total_mv)
                .bind(row.circ_mv)
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
                .bind(row.consecutive_up_days);
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

// --- Add these stubs near the top of your file ---

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
    ema_5: Option<f64>,
    ema_10: Option<f64>,
    ema_20: Option<f64>,
    ema_30: Option<f64>,
    ema_60: Option<f64>,
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
    williams_r_14: Option<f64>,
    aroon_up_25: Option<f64>,
    aroon_down_25: Option<f64>,
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
    ps: Option<f64>,
    ps_ttm: Option<f64>,
    dv_ratio: Option<f64>,
    dv_ttm: Option<f64>,
    total_share: Option<f64>,
    float_share: Option<f64>,
    free_share: Option<f64>,
    total_mv: Option<f64>,
    circ_mv: Option<f64>,

    // NEW: 5 predictive features for accuracy improvement
    pe_percentile_52w: Option<f64>,
    sector_momentum_vs_market: Option<f64>,
    volume_accel_5d: Option<f64>,
    price_vs_52w_high: Option<f64>,
    consecutive_up_days: Option<i32>,
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
    ps: f64,
    ps_ttm: f64,
    dv_ratio: f64,
    dv_ttm: f64,
    total_share: f64,
    float_share: f64,
    free_share: f64,
    total_mv: f64,
    circ_mv: f64,
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
            COALESCE(ps::DOUBLE PRECISION, 0.0) as "ps!",
            COALESCE(ps_ttm::DOUBLE PRECISION, 0.0) as "ps_ttm!",
            COALESCE(dv_ratio::DOUBLE PRECISION, 0.0) as "dv_ratio!",
            COALESCE(dv_ttm::DOUBLE PRECISION, 0.0) as "dv_ttm!",
            COALESCE(total_share::DOUBLE PRECISION, 0.0) as "total_share!",
            COALESCE(float_share::DOUBLE PRECISION, 0.0) as "float_share!",
            COALESCE(free_share::DOUBLE PRECISION, 0.0) as "free_share!",
            COALESCE(total_mv::DOUBLE PRECISION, 0.0) as "total_mv!",
            COALESCE(circ_mv::DOUBLE PRECISION, 0.0) as "circ_mv!"
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
                ps: row.ps,
                ps_ttm: row.ps_ttm,
                dv_ratio: row.dv_ratio,
                dv_ttm: row.dv_ttm,
                total_share: row.total_share,
                float_share: row.float_share,
                free_share: row.free_share,
                total_mv: row.total_mv,
                circ_mv: row.circ_mv,
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

    // CRITICAL FIX: Skip stocks with insufficient historical data
    // We need at least 60 days of data before attempting to calculate technical indicators
    // that require 20+ day lookback periods (Bollinger Bands, etc.)
    let min_history_required = 60;
    if daily_data.len() < min_history_required {
        eprintln!(
            "⚠️  Stock {} has only {} days of history (need {}). Skipping feature calculation.",
            ts_code,
            daily_data.len(),
            min_history_required
        );
        return vec![];
    }

    // --- Prefetch daily_basic and moneyflow data for this stock (with historical buffer) ---
    let daily_basic_map = prefetch_daily_basic_map(pool, ts_code, &fetch_start, max_date).await;
    let moneyflow_map = prefetch_moneyflow_map(pool, ts_code, &fetch_start, max_date).await;

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

        // Moving averages
        let ema_5 = if closes.len() >= 5 {
            Some(calculate_ema_custom(&closes, 5))
        } else {
            None
        };
        let ema_10 = if closes.len() >= 10 {
            Some(calculate_ema_custom(&closes, 10))
        } else {
            None
        };
        let ema_20 = if closes.len() >= 20 {
            Some(calculate_ema_custom(&closes, 20))
        } else {
            None
        };
        let ema_30 = if closes.len() >= 30 {
            Some(calculate_ema_custom(&closes, 30))
        } else {
            None
        };
        let ema_60 = if closes.len() >= 60 {
            Some(calculate_ema_custom(&closes, 60))
        } else {
            None
        };
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


        // Aroon
        let (aroon_up_25, aroon_down_25) = if highs.len() >= 25 {
            let (up, down) = calculate_aroon(&highs, &lows, 25);
            (Some(up), Some(down))
        } else {
            (None, None)
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
            if r > 0.002 {
                1
            } else if r < -0.002 {
                -1
            } else {
                0
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
            if r > 0.005 {
                1
            } else if r < -0.005 {
                -1
            } else {
                0
            }
        });

        // --- Map daily_basic fields by (ts_code, trade_date) ---
        let daily_basic = daily_basic_map.get(&(ts_code.to_string(), day.trade_date.clone()));

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

        // Williams %R
        let williams_r_14 = if highs.len() >= 14 {
            Some(calculate_williams_r(&highs, &lows, day.close, 14))
        } else {
            None
        };

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
        let moneyflow = moneyflow_map.get(&(ts_code.to_string(), day.trade_date.clone()));

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
                    moneyflow_map.get(&(ts_code.to_string(), daily_data[j].trade_date.clone()))
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
                ema_5,
                ema_10,
                ema_20,
                ema_30,
                ema_60,
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
                williams_r_14,
                aroon_up_25,
                aroon_down_25,
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
                ps: daily_basic.as_ref().map(|db| db.ps),
                ps_ttm: daily_basic.as_ref().map(|db| db.ps_ttm),
                dv_ratio: daily_basic.as_ref().map(|db| db.dv_ratio),
                dv_ttm: Some(daily_basic.as_ref().map(|db| db.dv_ttm).unwrap_or(0.0)), // Default to 0.0 if no dividends
                total_share: daily_basic.as_ref().map(|db| db.total_share),
                float_share: daily_basic.as_ref().map(|db| db.float_share),
                free_share: daily_basic.as_ref().map(|db| db.free_share),
                total_mv: daily_basic.as_ref().map(|db| db.total_mv),
                circ_mv: daily_basic.as_ref().map(|db| db.circ_mv),
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
            });
        } // end if day.trade_date >= min_date
    }
    // Runtime guard: drop any feature rows with trade_date after current_date
    let total_rows = features.len();
    let filtered: Vec<FeatureRow> = features
        .into_iter()
        .filter(|r| r.trade_date.as_str() <= current_date)
        .collect();
    let dropped = total_rows.saturating_sub(filtered.len());
    if dropped > 0 {
        eprintln!("⚠️  Dropped {} future-dated feature rows (CURRENT_DATE={}) for {}", dropped, current_date, ts_code);
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
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS ps DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS ps_ttm DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS dv_ratio DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS dv_ttm DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS total_share DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS float_share DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS free_share DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS total_mv DOUBLE PRECISION;",
        "ALTER TABLE ml_training_dataset ADD COLUMN IF NOT EXISTS circ_mv DOUBLE PRECISION;",
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
    ];

    // Run CREATE with PRIMARY KEY
    sqlx::query(
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
            ema_5 DOUBLE PRECISION,
            ema_10 DOUBLE PRECISION,
            ema_20 DOUBLE PRECISION,
            ema_30 DOUBLE PRECISION,
            ema_60 DOUBLE PRECISION,
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
            volume_ratio DOUBLE PRECISION, -- feature column
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
            williams_r_14 DOUBLE PRECISION,
            aroon_up_25 DOUBLE PRECISION,
            aroon_down_25 DOUBLE PRECISION,
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
           
            -- DailyBasic columns (excluding close)
            turnover_rate DOUBLE PRECISION,
            turnover_rate_f DOUBLE PRECISION,
            -- DO NOT add volume_ratio again here!
            pe DOUBLE PRECISION,
            pe_ttm DOUBLE PRECISION,
            pb DOUBLE PRECISION,
            ps DOUBLE PRECISION,
            ps_ttm DOUBLE PRECISION,
            dv_ratio DOUBLE PRECISION,
            dv_ttm DOUBLE PRECISION,
            total_share DOUBLE PRECISION,
            float_share DOUBLE PRECISION,
            free_share DOUBLE PRECISION,
            total_mv DOUBLE PRECISION,
            circ_mv DOUBLE PRECISION,
            vol_percentile DOUBLE PRECISION,
            high_vol_regime SMALLINT,
            next_day_return DOUBLE PRECISION,
            next_day_direction SMALLINT,
            next_3day_return DOUBLE PRECISION,
            next_3day_direction SMALLINT
        );
        "#,
    )
    .execute(pool)
    .await?;

    // Run ALTER TABLEs to add missing columns (idempotent, safe if already exist)
    for stmt in alter_statements.iter() {
        sqlx::query(stmt).execute(pool).await.ok();
    }

    Ok(())
}
