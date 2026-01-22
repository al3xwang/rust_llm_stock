/// Trading signal generator: Converts predictions into actionable signals stored in database
///
/// Reads from: stock_predictions table (predictions from batch_predict)
/// Writes to: trading_signals table
///
/// Features:
/// - Combines model predictions with technical indicators
/// - Applies filtering (confidence, volume, market cap, industry)
/// - Generates BUY/SELL signals with composite scores
/// - Stores signals in database for downstream consumption
///
/// Usage:
///   cargo run --release --bin generate_signals
///   cargo run --release --bin generate_signals -- --date 20260115 --min-confidence 0.7
///   cargo run --release --bin generate_signals -- --show --industries "银行,保险"
use anyhow::Result;
use clap::Parser;
use rust_llm_stock::stock_db::get_connection;
use sqlx::{Pool, Postgres, Row};

#[derive(Parser)]
#[command(name = "generate-signals")]
#[command(about = "Generate buy/sell signals from stock predictions and store in database")]
struct Cli {
    /// Trade date to generate signals for (YYYYMMDD format, default: latest)
    #[arg(short, long)]
    date: Option<String>,

    /// Minimum confidence threshold for signals (0.0-1.0)
    #[arg(long, default_value_t = 0.6)]
    min_confidence: f64,

    /// Minimum predicted return for BUY signals (percentage)
    #[arg(long, default_value_t = 0.5)]
    min_buy_return: f64,

    /// Maximum predicted return for SELL signals (percentage, negative)
    #[arg(long, default_value_t = -0.5)]
    max_sell_return: f64,

    /// Use 3-day predictions instead of 1-day
    #[arg(long)]
    use_3day: bool,

    /// Minimum volume ratio (relative to 5-day average)
    #[arg(long, default_value_t = 0.8)]
    min_volume_ratio: f64,

    /// Model version identifier
    #[arg(long, default_value = "v1")]
    model_version: String,

    /// Filter by industry (comma-separated list)
    #[arg(long)]
    industries: Option<String>,

    /// Exclude ST stocks
    #[arg(long, default_value_t = true)]
    exclude_st: bool,

    /// Minimum market cap in billions (total_mv)
    #[arg(long)]
    min_market_cap: Option<f64>,

    /// Maximum market cap in billions
    #[arg(long)]
    max_market_cap: Option<f64>,

    /// Show signals after generation (table output)
    #[arg(long)]
    show: bool,

    /// Delete existing signals for the date before generating new ones
    #[arg(long)]
    replace: bool,

    /// Dry run - don't insert into database
    #[arg(long)]
    dry_run: bool,

    /// Query existing signals from database
    #[arg(long)]
    query: bool,

    /// Limit number of signals to display when querying
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Debug, Clone)]
struct Signal {
    ts_code: String,
    trade_date: String,
    signal_type: SignalType,
    score: f64,
    predicted_return_1day: f64,
    predicted_return_3day: Option<f64>,
    confidence_1day: f64,
    confidence_3day: Option<f64>,
    // Technical context
    rsi_14: Option<f64>,
    macd_histogram: Option<f64>,
    bb_percent_b: Option<f64>,
    volume_ratio: Option<f64>,
    price_momentum_5: Option<f64>,
    adx_14: Option<f64>,
    // Fundamental
    total_mv: Option<f64>,
    pe_ttm: Option<f64>,
    pb: Option<f64>,
    // Info
    stock_name: Option<String>,
    industry: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum SignalType {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

impl SignalType {
    fn as_str(&self) -> &'static str {
        match self {
            SignalType::StrongBuy => "STRONG_BUY",
            SignalType::Buy => "BUY",
            SignalType::Hold => "HOLD",
            SignalType::Sell => "SELL",
            SignalType::StrongSell => "STRONG_SELL",
        }
    }

    fn emoji(&self) -> &'static str {
        match self {
            SignalType::StrongBuy => "🚀",
            SignalType::Buy => "📈",
            SignalType::Hold => "➖",
            SignalType::Sell => "📉",
            SignalType::StrongSell => "⚠️",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "STRONG_BUY" => SignalType::StrongBuy,
            "BUY" => SignalType::Buy,
            "SELL" => SignalType::Sell,
            "STRONG_SELL" => SignalType::StrongSell,
            _ => SignalType::Hold,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Stock Signal Generation System                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Database connection
    println!("Connecting to database...");
    let pool = get_connection().await;
    println!("✓ Database connected!");
    println!();

    // Ensure table exists
    create_signals_table(&pool).await?;

    // Query mode: just display existing signals
    if cli.query {
        let trade_date = match &cli.date {
            Some(d) => d.clone(),
            None => get_latest_signal_date(&pool)
                .await
                .unwrap_or_else(|_| "".to_string()),
        };
        if trade_date.is_empty() {
            println!("No signals found in database.");
            return Ok(());
        }
        query_and_display_signals(&pool, &trade_date, &cli).await?;
        return Ok(());
    }

    // Determine trade date
    let trade_date = match &cli.date {
        Some(d) => d.clone(),
        None => get_latest_prediction_date(&pool).await?,
    };

    println!("Generating signals for: {}", trade_date);
    println!("Model version: {}", cli.model_version);
    println!();

    // Fetch predictions for the date
    let predictions = fetch_predictions_with_context(&pool, &trade_date).await?;
    println!("Found {} predictions for {}", predictions.len(), trade_date);

    if predictions.is_empty() {
        println!(
            "No predictions found for {}. Run batch_predict first.",
            trade_date
        );
        return Ok(());
    }

    // Parse industry filter
    let industry_filter: Option<Vec<String>> = cli
        .industries
        .as_ref()
        .map(|s| s.split(',').map(|i| i.trim().to_string()).collect());

    // Generate signals
    let signals = generate_signals(&predictions, &cli, &industry_filter);

    println!("Generated {} signals", signals.len());

    if signals.is_empty() {
        println!("No signals met the criteria.");
        return Ok(());
    }

    // Delete existing signals if --replace
    if cli.replace && !cli.dry_run {
        let deleted = delete_signals_for_date(&pool, &trade_date, &cli.model_version).await?;
        println!(
            "Deleted {} existing signals for {} (model: {})",
            deleted, trade_date, cli.model_version
        );
    }

    // Insert signals into database
    if !cli.dry_run {
        let (inserted, updated) = insert_signals(&pool, &signals, &cli.model_version).await?;
        println!(
            "✓ Inserted {} new signals, updated {} existing",
            inserted, updated
        );
    } else {
        println!("⚠️  Dry run - no signals inserted");
    }

    // Show signals if requested
    if cli.show || cli.dry_run {
        display_signals(&signals, &trade_date);
    }

    // Print summary
    print_summary(&signals);

    Ok(())
}

async fn create_signals_table(pool: &Pool<Postgres>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS trading_signals (
            id SERIAL PRIMARY KEY,
            ts_code VARCHAR(20) NOT NULL,
            trade_date VARCHAR(8) NOT NULL,
            signal_type VARCHAR(20) NOT NULL,
            signal_score DOUBLE PRECISION NOT NULL,
            predicted_return_1day DOUBLE PRECISION,
            predicted_return_3day DOUBLE PRECISION,
            confidence_1day DOUBLE PRECISION,
            confidence_3day DOUBLE PRECISION,
            rsi_14 DOUBLE PRECISION,
            macd_histogram DOUBLE PRECISION,
            bb_percent_b DOUBLE PRECISION,
            volume_ratio DOUBLE PRECISION,
            price_momentum_5 DOUBLE PRECISION,
            adx_14 DOUBLE PRECISION,
            total_mv DOUBLE PRECISION,
            pe_ttm DOUBLE PRECISION,
            pb DOUBLE PRECISION,
            stock_name VARCHAR(100),
            industry VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            model_version VARCHAR(50),
            actual_return_1day DOUBLE PRECISION,
            actual_return_3day DOUBLE PRECISION,
            signal_correct_1day BOOLEAN,
            signal_correct_3day BOOLEAN,
            CONSTRAINT uq_signal_stock_date_model UNIQUE (ts_code, trade_date, model_version)
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Create indexes
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_signals_trade_date ON trading_signals(trade_date)")
        .execute(pool)
        .await
        .ok();
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON trading_signals(signal_type)",
    )
    .execute(pool)
    .await
    .ok();
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_signals_score ON trading_signals(signal_score DESC)",
    )
    .execute(pool)
    .await
    .ok();

    Ok(())
}

async fn get_latest_prediction_date(pool: &Pool<Postgres>) -> Result<String> {
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT MAX(trade_date) FROM stock_predictions WHERE predicted_return IS NOT NULL",
    )
    .fetch_one(pool)
    .await?;

    row.0
        .ok_or_else(|| anyhow::anyhow!("No predictions found in database"))
}

async fn get_latest_signal_date(pool: &Pool<Postgres>) -> Result<String> {
    let row: (Option<String>,) = sqlx::query_as("SELECT MAX(trade_date) FROM trading_signals")
        .fetch_one(pool)
        .await?;

    row.0
        .ok_or_else(|| anyhow::anyhow!("No signals found in database"))
}

#[derive(Debug)]
#[allow(dead_code)]
struct PredictionWithContext {
    ts_code: String,
    trade_date: String,
    predicted_return: f64,
    predicted_direction: bool,
    confidence: f64,
    predicted_3day_return: Option<f64>,
    predicted_3day_direction: Option<bool>,
    // Stock info
    name: Option<String>,
    industry: Option<String>,
    // Technical indicators
    rsi_14: Option<f64>,
    macd_histogram: Option<f64>,
    bb_percent_b: Option<f64>,
    volume_ratio: Option<f64>,
    price_momentum_5: Option<f64>,
    adx_14: Option<f64>,
    // Fundamental data
    total_mv: Option<f64>,
    pe_ttm: Option<f64>,
    pb: Option<f64>,
}

async fn fetch_predictions_with_context(
    pool: &Pool<Postgres>,
    trade_date: &str,
) -> Result<Vec<PredictionWithContext>> {
    let rows = sqlx::query(
        r#"
        SELECT 
            p.ts_code,
            p.trade_date,
            COALESCE(p.predicted_return, 0.0) as predicted_return,
            COALESCE(p.predicted_direction, false) as predicted_direction,
            COALESCE(p.confidence, 0.0) as confidence,
            p.predicted_3day_return,
            p.predicted_3day_direction,
            sb.name,
            sb.industry,
            m.rsi_14,
            m.macd_histogram,
            m.bb_percent_b,
            m.volume_ratio,
            m.price_momentum_5,
            m.adx_14,
            m.total_mv,
            m.pe_ttm,
            m.pb
        FROM stock_predictions p
        LEFT JOIN stock_basic sb ON p.ts_code = sb.ts_code
        LEFT JOIN ml_training_dataset m ON p.ts_code = m.ts_code AND p.trade_date = m.trade_date
        WHERE p.trade_date = $1
          AND p.predicted_return IS NOT NULL
        ORDER BY p.confidence DESC
        "#,
    )
    .bind(trade_date)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|row| PredictionWithContext {
            ts_code: row.get(0),
            trade_date: row.get(1),
            predicted_return: row.get(2),
            predicted_direction: row.get(3),
            confidence: row.get(4),
            predicted_3day_return: row.get(5),
            predicted_3day_direction: row.get(6),
            name: row.get(7),
            industry: row.get(8),
            rsi_14: row.get(9),
            macd_histogram: row.get(10),
            bb_percent_b: row.get(11),
            volume_ratio: row.get(12),
            price_momentum_5: row.get(13),
            adx_14: row.get(14),
            total_mv: row.get(15),
            pe_ttm: row.get(16),
            pb: row.get(17),
        })
        .collect())
}

fn generate_signals(
    predictions: &[PredictionWithContext],
    cli: &Cli,
    industry_filter: &Option<Vec<String>>,
) -> Vec<Signal> {
    let mut signals = Vec::new();

    for pred in predictions {
        // Filter: ST stocks
        if cli.exclude_st {
            if let Some(ref name) = pred.name {
                if name.contains("ST") || name.contains("*ST") {
                    continue;
                }
            }
        }

        // Filter: confidence
        if pred.confidence < cli.min_confidence {
            continue;
        }

        // Filter: volume ratio
        if let Some(vr) = pred.volume_ratio {
            if vr < cli.min_volume_ratio {
                continue;
            }
        }

        // Filter: industry
        if let Some(filter) = industry_filter {
            if let Some(ref industry) = pred.industry {
                if !filter.iter().any(|f| industry.contains(f)) {
                    continue;
                }
            } else {
                continue;
            }
        }

        // Filter: market cap
        if let Some(min_cap) = cli.min_market_cap {
            if let Some(mv) = pred.total_mv {
                if mv < min_cap * 10000.0 {
                    continue;
                }
            }
        }
        if let Some(max_cap) = cli.max_market_cap {
            if let Some(mv) = pred.total_mv {
                if mv > max_cap * 10000.0 {
                    continue;
                }
            }
        }

        // Determine return to use
        let predicted_return = if cli.use_3day {
            pred.predicted_3day_return.unwrap_or(pred.predicted_return)
        } else {
            pred.predicted_return
        };

        // Determine signal type
        let signal_type = determine_signal_type(
            predicted_return,
            pred.confidence,
            pred.rsi_14,
            pred.macd_histogram,
            pred.bb_percent_b,
            cli.min_buy_return,
            cli.max_sell_return,
        );

        // Skip HOLD signals
        if signal_type == SignalType::Hold {
            continue;
        }

        // Calculate composite score
        let score = calculate_signal_score(
            &signal_type,
            predicted_return,
            pred.confidence,
            pred.rsi_14,
            pred.macd_histogram,
            pred.volume_ratio,
            pred.adx_14,
        );

        signals.push(Signal {
            ts_code: pred.ts_code.clone(),
            trade_date: pred.trade_date.clone(),
            signal_type,
            score,
            predicted_return_1day: pred.predicted_return,
            predicted_return_3day: pred.predicted_3day_return,
            confidence_1day: pred.confidence,
            confidence_3day: None,
            rsi_14: pred.rsi_14,
            macd_histogram: pred.macd_histogram,
            bb_percent_b: pred.bb_percent_b,
            volume_ratio: pred.volume_ratio,
            price_momentum_5: pred.price_momentum_5,
            adx_14: pred.adx_14,
            total_mv: pred.total_mv,
            pe_ttm: pred.pe_ttm,
            pb: pred.pb,
            stock_name: pred.name.clone(),
            industry: pred.industry.clone(),
        });
    }

    // Sort by score descending
    signals.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    signals
}

fn determine_signal_type(
    predicted_return: f64,
    confidence: f64,
    rsi: Option<f64>,
    macd_hist: Option<f64>,
    bb_pct_b: Option<f64>,
    min_buy: f64,
    max_sell: f64,
) -> SignalType {
    let mut buy_bonus = 0;
    let mut sell_bonus = 0;

    // RSI signals
    if let Some(rsi) = rsi {
        if rsi < 30.0 {
            buy_bonus += 2;
        } else if rsi < 40.0 {
            buy_bonus += 1;
        } else if rsi > 70.0 {
            sell_bonus += 2;
        } else if rsi > 60.0 {
            sell_bonus += 1;
        }
    }

    // MACD histogram
    if let Some(macd) = macd_hist {
        if macd > 0.0 {
            buy_bonus += 1;
        } else if macd < 0.0 {
            sell_bonus += 1;
        }
    }

    // Bollinger Band position
    if let Some(bb) = bb_pct_b {
        if bb < 0.2 {
            buy_bonus += 1;
        } else if bb > 0.8 {
            sell_bonus += 1;
        }
    }

    if predicted_return >= min_buy * 2.0 && confidence >= 0.7 && buy_bonus >= 2 {
        SignalType::StrongBuy
    } else if predicted_return >= min_buy && confidence >= 0.6 {
        SignalType::Buy
    } else if predicted_return <= max_sell * 2.0 && confidence >= 0.7 && sell_bonus >= 2 {
        SignalType::StrongSell
    } else if predicted_return <= max_sell && confidence >= 0.6 {
        SignalType::Sell
    } else {
        SignalType::Hold
    }
}

fn calculate_signal_score(
    signal_type: &SignalType,
    predicted_return: f64,
    confidence: f64,
    rsi: Option<f64>,
    macd_hist: Option<f64>,
    volume_ratio: Option<f64>,
    adx: Option<f64>,
) -> f64 {
    let return_score = predicted_return.abs() * 10.0;
    let confidence_score = confidence * 100.0;
    let mut tech_score = 0.0;

    if let Some(rsi) = rsi {
        match signal_type {
            SignalType::StrongBuy | SignalType::Buy => {
                if rsi < 40.0 {
                    tech_score += (40.0 - rsi) * 0.5;
                }
            }
            SignalType::StrongSell | SignalType::Sell => {
                if rsi > 60.0 {
                    tech_score += (rsi - 60.0) * 0.5;
                }
            }
            _ => {}
        }
    }

    if let Some(macd) = macd_hist {
        match signal_type {
            SignalType::StrongBuy | SignalType::Buy if macd > 0.0 => tech_score += 5.0,
            SignalType::StrongSell | SignalType::Sell if macd < 0.0 => tech_score += 5.0,
            _ => {}
        }
    }

    if let Some(vr) = volume_ratio {
        if vr > 1.5 {
            tech_score += 10.0;
        } else if vr > 1.0 {
            tech_score += 5.0;
        }
    }

    if let Some(adx) = adx {
        if adx > 25.0 {
            tech_score += (adx - 25.0) * 0.2;
        }
    }

    (return_score * 0.3 + confidence_score * 0.5 + tech_score * 0.2)
        .min(100.0)
        .max(0.0)
}

async fn delete_signals_for_date(
    pool: &Pool<Postgres>,
    trade_date: &str,
    model_version: &str,
) -> Result<u64> {
    let result =
        sqlx::query("DELETE FROM trading_signals WHERE trade_date = $1 AND model_version = $2")
            .bind(trade_date)
            .bind(model_version)
            .execute(pool)
            .await?;

    Ok(result.rows_affected())
}

async fn insert_signals(
    pool: &Pool<Postgres>,
    signals: &[Signal],
    model_version: &str,
) -> Result<(usize, usize)> {
    let mut inserted = 0;
    let mut updated = 0;

    for signal in signals {
        let result = sqlx::query(
            r#"
            INSERT INTO trading_signals (
                ts_code, trade_date, signal_type, signal_score,
                predicted_return_1day, predicted_return_3day,
                confidence_1day, confidence_3day,
                rsi_14, macd_histogram, bb_percent_b, volume_ratio,
                price_momentum_5, adx_14,
                total_mv, pe_ttm, pb,
                stock_name, industry, model_version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
            )
            ON CONFLICT (ts_code, trade_date, model_version)
            DO UPDATE SET
                signal_type = EXCLUDED.signal_type,
                signal_score = EXCLUDED.signal_score,
                predicted_return_1day = EXCLUDED.predicted_return_1day,
                predicted_return_3day = EXCLUDED.predicted_return_3day,
                confidence_1day = EXCLUDED.confidence_1day,
                confidence_3day = EXCLUDED.confidence_3day,
                rsi_14 = EXCLUDED.rsi_14,
                macd_histogram = EXCLUDED.macd_histogram,
                bb_percent_b = EXCLUDED.bb_percent_b,
                volume_ratio = EXCLUDED.volume_ratio,
                price_momentum_5 = EXCLUDED.price_momentum_5,
                adx_14 = EXCLUDED.adx_14,
                total_mv = EXCLUDED.total_mv,
                pe_ttm = EXCLUDED.pe_ttm,
                pb = EXCLUDED.pb,
                stock_name = EXCLUDED.stock_name,
                industry = EXCLUDED.industry,
                created_at = NOW()
            RETURNING (xmax = 0) AS was_inserted
            "#,
        )
        .bind(&signal.ts_code)
        .bind(&signal.trade_date)
        .bind(signal.signal_type.as_str())
        .bind(signal.score)
        .bind(signal.predicted_return_1day)
        .bind(signal.predicted_return_3day)
        .bind(signal.confidence_1day)
        .bind(signal.confidence_3day)
        .bind(signal.rsi_14)
        .bind(signal.macd_histogram)
        .bind(signal.bb_percent_b)
        .bind(signal.volume_ratio)
        .bind(signal.price_momentum_5)
        .bind(signal.adx_14)
        .bind(signal.total_mv)
        .bind(signal.pe_ttm)
        .bind(signal.pb)
        .bind(&signal.stock_name)
        .bind(&signal.industry)
        .bind(model_version)
        .fetch_one(pool)
        .await?;

        let was_inserted: bool = result.get("was_inserted");
        if was_inserted {
            inserted += 1;
        } else {
            updated += 1;
        }
    }

    Ok((inserted, updated))
}

async fn query_and_display_signals(
    pool: &Pool<Postgres>,
    trade_date: &str,
    cli: &Cli,
) -> Result<()> {
    println!("Querying signals for date: {}", trade_date);
    println!();

    let rows = sqlx::query(
        r#"
        SELECT 
            ts_code, trade_date, signal_type, signal_score,
            predicted_return_1day, predicted_return_3day,
            confidence_1day, rsi_14, stock_name, industry
        FROM trading_signals
        WHERE trade_date = $1
        ORDER BY signal_score DESC
        LIMIT $2
        "#,
    )
    .bind(trade_date)
    .bind(cli.limit as i64)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        println!("No signals found for {}", trade_date);
        return Ok(());
    }

    let signals: Vec<Signal> = rows
        .iter()
        .map(|row| {
            let signal_type_str: String = row.get(2);
            Signal {
                ts_code: row.get(0),
                trade_date: row.get(1),
                signal_type: SignalType::from_str(&signal_type_str),
                score: row.get(3),
                predicted_return_1day: row.get::<Option<f64>, _>(4).unwrap_or(0.0),
                predicted_return_3day: row.get(5),
                confidence_1day: row.get::<Option<f64>, _>(6).unwrap_or(0.0),
                confidence_3day: None,
                rsi_14: row.get(7),
                macd_histogram: None,
                bb_percent_b: None,
                volume_ratio: None,
                price_momentum_5: None,
                adx_14: None,
                total_mv: None,
                pe_ttm: None,
                pb: None,
                stock_name: row.get(8),
                industry: row.get(9),
            }
        })
        .collect();

    display_signals(&signals, trade_date);
    print_summary(&signals);

    Ok(())
}

fn display_signals(signals: &[Signal], trade_date: &str) {
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              Signals for {}                                          ║",
        trade_date
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Group by signal type
    let buy_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::StrongBuy | SignalType::Buy))
        .collect();
    let sell_signals: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::StrongSell | SignalType::Sell))
        .collect();

    if !buy_signals.is_empty() {
        println!();
        println!(
            "┌─────────────────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "│  📈 BUY SIGNALS ({})                                                                        │",
            buy_signals.len()
        );
        println!(
            "├──────────┬────────────────────┬────────────┬───────────┬────────┬────────┬────────┬─────────┤"
        );
        println!(
            "│ Code     │ Name               │ Signal     │ Pred Ret  │ Conf   │ RSI    │ Score  │ Industry│"
        );
        println!(
            "├──────────┼────────────────────┼────────────┼───────────┼────────┼────────┼────────┼─────────┤"
        );

        for signal in buy_signals.iter().take(25) {
            let name = signal
                .stock_name
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(18)
                .collect::<String>();
            let industry = signal
                .industry
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(7)
                .collect::<String>();
            println!(
                "│ {:8} │ {:18} │ {:10} │ {:+8.2}% │ {:5.1}% │ {:6.1} │ {:6.1} │ {:7} │",
                signal.ts_code,
                name,
                format!(
                    "{} {}",
                    signal.signal_type.emoji(),
                    signal.signal_type.as_str()
                ),
                signal.predicted_return_1day,
                signal.confidence_1day * 100.0,
                signal.rsi_14.unwrap_or(0.0),
                signal.score,
                industry,
            );
        }
        println!(
            "└──────────┴────────────────────┴────────────┴───────────┴────────┴────────┴────────┴─────────┘"
        );
    }

    if !sell_signals.is_empty() {
        println!();
        println!(
            "┌─────────────────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "│  📉 SELL SIGNALS ({})                                                                       │",
            sell_signals.len()
        );
        println!(
            "├──────────┬────────────────────┬────────────┬───────────┬────────┬────────┬────────┬─────────┤"
        );
        println!(
            "│ Code     │ Name               │ Signal     │ Pred Ret  │ Conf   │ RSI    │ Score  │ Industry│"
        );
        println!(
            "├──────────┼────────────────────┼────────────┼───────────┼────────┼────────┼────────┼─────────┤"
        );

        for signal in sell_signals.iter().take(25) {
            let name = signal
                .stock_name
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(18)
                .collect::<String>();
            let industry = signal
                .industry
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(7)
                .collect::<String>();
            println!(
                "│ {:8} │ {:18} │ {:10} │ {:+8.2}% │ {:5.1}% │ {:6.1} │ {:6.1} │ {:7} │",
                signal.ts_code,
                name,
                format!(
                    "{} {}",
                    signal.signal_type.emoji(),
                    signal.signal_type.as_str()
                ),
                signal.predicted_return_1day,
                signal.confidence_1day * 100.0,
                signal.rsi_14.unwrap_or(0.0),
                signal.score,
                industry,
            );
        }
        println!(
            "└──────────┴────────────────────┴────────────┴───────────┴────────┴────────┴────────┴─────────┘"
        );
    }

    println!();
}

fn print_summary(signals: &[Signal]) {
    let strong_buys = signals
        .iter()
        .filter(|s| s.signal_type == SignalType::StrongBuy)
        .count();
    let buys = signals
        .iter()
        .filter(|s| s.signal_type == SignalType::Buy)
        .count();
    let sells = signals
        .iter()
        .filter(|s| s.signal_type == SignalType::Sell)
        .count();
    let strong_sells = signals
        .iter()
        .filter(|s| s.signal_type == SignalType::StrongSell)
        .count();

    let avg_buy_return = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::StrongBuy | SignalType::Buy))
        .map(|s| s.predicted_return_1day)
        .sum::<f64>()
        / (strong_buys + buys).max(1) as f64;

    let avg_sell_return = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::StrongSell | SignalType::Sell))
        .map(|s| s.predicted_return_1day)
        .sum::<f64>()
        / (sells + strong_sells).max(1) as f64;

    let avg_confidence =
        signals.iter().map(|s| s.confidence_1day).sum::<f64>() / signals.len().max(1) as f64;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    Signal Summary                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Total Signals:     {}", signals.len());
    println!();
    println!("  🚀 Strong Buy:     {}", strong_buys);
    println!("  📈 Buy:            {}", buys);
    println!("  📉 Sell:           {}", sells);
    println!("  ⚠️  Strong Sell:   {}", strong_sells);
    println!();
    println!("  Avg Buy Return:    {:+.2}%", avg_buy_return);
    println!("  Avg Sell Return:   {:+.2}%", avg_sell_return);
    println!("  Avg Confidence:    {:.1}%", avg_confidence * 100.0);
    println!();

    // Top picks
    let top_buys: Vec<_> = signals
        .iter()
        .filter(|s| matches!(s.signal_type, SignalType::StrongBuy | SignalType::Buy))
        .take(5)
        .collect();

    if !top_buys.is_empty() {
        println!("  🏆 Top 5 Buy Picks:");
        for (i, signal) in top_buys.iter().enumerate() {
            println!(
                "     {}. {} ({}) - {:+.2}% @ {:.0}% confidence",
                i + 1,
                signal.ts_code,
                signal.stock_name.as_deref().unwrap_or("-"),
                signal.predicted_return_1day,
                signal.confidence_1day * 100.0
            );
        }
    }

    println!();
}
