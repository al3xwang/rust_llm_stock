/// Backtesting engine for trading signals
///
/// Simulates trading based on historical signals and calculates performance metrics.
///
/// Usage:
///   cargo run --release --bin backtest_signals
///   cargo run --release --bin backtest_signals -- --start-date 20240101 --end-date 20241231
///   cargo run --release --bin backtest_signals -- --initial-capital 1000000 --max-positions 10
///   cargo run --release --bin backtest_signals -- --from-predictions --long-only
use anyhow::Result;
use clap::Parser;
use rust_llm_stock::stock_db::get_connection;
use sqlx::{Pool, Postgres, Row};
use std::collections::HashMap;

/// Format a number with thousands separators
fn format_number(val: f64) -> String {
    let s = format!("{:.0}", val.abs());
    let chars: Vec<char> = s.chars().rev().collect();
    let chunks: Vec<String> = chars
        .chunks(3)
        .map(|c| c.iter().collect::<String>())
        .collect();
    let formatted = chunks.join(",").chars().rev().collect::<String>();
    if val < 0.0 {
        format!("-{}", formatted)
    } else {
        formatted
    }
}

fn format_money(val: f64) -> String {
    format!("¥{}", format_number(val))
}

fn format_money_signed(val: f64) -> String {
    if val >= 0.0 {
        format!("+¥{}", format_number(val))
    } else {
        format!("-¥{}", format_number(val.abs()))
    }
}

fn format_pnl(val: f64) -> String {
    if val >= 0.0 {
        format!("+{:.0}", val)
    } else {
        format!("{:.0}", val)
    }
}

#[derive(Parser)]
#[command(name = "backtest-signals")]
#[command(about = "Backtest trading signals against historical market data")]
struct Cli {
    /// Start date for backtest (YYYYMMDD)
    #[arg(long)]
    start_date: Option<String>,

    /// End date for backtest (YYYYMMDD)
    #[arg(long)]
    end_date: Option<String>,

    /// Initial capital in CNY
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_capital: f64,

    /// Maximum number of positions to hold at once
    #[arg(long, default_value_t = 10)]
    max_positions: usize,

    /// Position size as percentage of capital (0.0-1.0)
    #[arg(long, default_value_t = 0.1)]
    position_size: f64,

    /// Minimum signal score to trade
    #[arg(long, default_value_t = 50.0)]
    min_score: f64,

    /// Minimum confidence to trade
    #[arg(long, default_value_t = 0.6)]
    min_confidence: f64,

    /// Only backtest BUY signals (long only)
    #[arg(long)]
    long_only: bool,

    /// Transaction cost per trade (percentage)
    #[arg(long, default_value_t = 0.001)]
    transaction_cost: f64,

    /// Use 3-day holding period instead of 1-day
    #[arg(long)]
    use_3day: bool,

    /// Model version to backtest
    #[arg(long, default_value = "v1")]
    model_version: String,

    /// Show detailed trade log
    #[arg(long)]
    verbose: bool,

    /// Generate signals from predictions if no signals exist
    #[arg(long)]
    from_predictions: bool,
}

#[derive(Debug, Clone)]
struct Trade {
    ts_code: String,
    entry_date: String,
    signal_type: String,
    signal_score: f64,
    confidence: f64,
    predicted_return: f64,
    actual_return: f64,
    pnl: f64,
    is_winner: bool,
}

#[derive(Debug, Clone)]
struct DailyPortfolio {
    date: String,
    capital: f64,
    positions: usize,
    cumulative_return: f64,
}

#[derive(Debug)]
struct BacktestResult {
    trades: Vec<Trade>,
    daily_portfolio: Vec<DailyPortfolio>,
    total_return: f64,
    annualized_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    profit_factor: f64,
    total_trades: usize,
    winning_trades: usize,
    losing_trades: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              Signal Backtesting Engine                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    println!("Connecting to database...");
    let pool = get_connection().await;
    println!("✓ Database connected!");
    println!();

    let (start_date, end_date) = get_date_range(&pool, &cli).await?;
    println!("Backtest period: {} to {}", start_date, end_date);
    println!("Initial capital: {}", format_money(cli.initial_capital));
    println!("Max positions: {}", cli.max_positions);
    println!("Position size: {:.0}%", cli.position_size * 100.0);
    println!(
        "Holding period: {}",
        if cli.use_3day { "3 days" } else { "1 day" }
    );
    println!();

    let signals = if cli.from_predictions {
        fetch_signals_from_predictions(&pool, &start_date, &end_date, &cli).await?
    } else {
        fetch_signals_with_returns(&pool, &start_date, &end_date, &cli).await?
    };

    if signals.is_empty() {
        println!("No signals found for the specified period.");
        println!(
            "Run generate_signals first, or use --from-predictions to backtest directly from predictions."
        );
        return Ok(());
    }

    println!("Found {} signals to backtest", signals.len());
    println!();

    let result = run_backtest(&signals, &cli);
    display_results(&result, &cli);

    if !cli.from_predictions {
        update_signal_actuals(&pool, &result.trades).await?;
    }

    Ok(())
}

async fn get_date_range(pool: &Pool<Postgres>, cli: &Cli) -> Result<(String, String)> {
    let start = match &cli.start_date {
        Some(d) => d.clone(),
        None => {
            let row: (Option<String>,) = sqlx::query_as(
                "SELECT MIN(trade_date) FROM stock_predictions WHERE predicted_return IS NOT NULL",
            )
            .fetch_one(pool)
            .await?;
            row.0.unwrap_or_else(|| "20240101".to_string())
        }
    };

    let end = match &cli.end_date {
        Some(d) => d.clone(),
        None => {
            let row: (Option<String>,) = sqlx::query_as(
                "SELECT MAX(trade_date) FROM stock_predictions WHERE predicted_return IS NOT NULL",
            )
            .fetch_one(pool)
            .await?;
            row.0.unwrap_or_else(|| "20251231".to_string())
        }
    };

    Ok((start, end))
}

#[derive(Debug)]
struct SignalWithReturn {
    ts_code: String,
    trade_date: String,
    signal_type: String,
    signal_score: f64,
    confidence: f64,
    predicted_return: f64,
    actual_return_1day: Option<f64>,
    actual_return_3day: Option<f64>,
}

async fn fetch_signals_with_returns(
    pool: &Pool<Postgres>,
    start_date: &str,
    end_date: &str,
    cli: &Cli,
) -> Result<Vec<SignalWithReturn>> {
    let mut query = String::from(
        r#"
        SELECT 
            s.ts_code,
            s.trade_date,
            s.signal_type,
            s.signal_score,
            COALESCE(s.confidence_1day, 0.0) as confidence,
            COALESCE(s.predicted_return_1day, 0.0) as predicted_return,
            m.next_day_return as actual_return_1day,
            m.next_3day_return as actual_return_3day
        FROM trading_signals s
        LEFT JOIN ml_training_dataset m ON s.ts_code = m.ts_code AND s.trade_date = m.trade_date
        WHERE s.trade_date >= $1 AND s.trade_date <= $2
          AND s.signal_score >= $3
          AND COALESCE(s.confidence_1day, 0.0) >= $4
        "#,
    );

    if cli.long_only {
        query.push_str(" AND s.signal_type IN ('STRONG_BUY', 'BUY')");
    }

    query.push_str(" ORDER BY s.trade_date, s.signal_score DESC");

    let rows = sqlx::query(&query)
        .bind(start_date)
        .bind(end_date)
        .bind(cli.min_score)
        .bind(cli.min_confidence)
        .fetch_all(pool)
        .await?;

    Ok(rows
        .iter()
        .map(|row| SignalWithReturn {
            ts_code: row.get(0),
            trade_date: row.get(1),
            signal_type: row.get(2),
            signal_score: row.get(3),
            confidence: row.get(4),
            predicted_return: row.get(5),
            actual_return_1day: row.get(6),
            actual_return_3day: row.get(7),
        })
        .collect())
}

async fn fetch_signals_from_predictions(
    pool: &Pool<Postgres>,
    start_date: &str,
    end_date: &str,
    cli: &Cli,
) -> Result<Vec<SignalWithReturn>> {
    println!("Fetching signals directly from predictions...");

    let mut query = String::from(
        r#"
        SELECT 
            p.ts_code,
            p.trade_date,
            CASE 
                WHEN p.predicted_return >= 1.0 AND p.confidence >= 0.7 THEN 'STRONG_BUY'
                WHEN p.predicted_return >= 0.5 AND p.confidence >= 0.6 THEN 'BUY'
                WHEN p.predicted_return <= -1.0 AND p.confidence >= 0.7 THEN 'STRONG_SELL'
                WHEN p.predicted_return <= -0.5 AND p.confidence >= 0.6 THEN 'SELL'
                ELSE 'HOLD'
            END as signal_type,
            (ABS(p.predicted_return) * 10 + p.confidence * 50) as signal_score,
            COALESCE(p.confidence, 0.0) as confidence,
            COALESCE(p.predicted_return, 0.0) as predicted_return,
            m.next_day_return as actual_return_1day,
            m.next_3day_return as actual_return_3day
        FROM stock_predictions p
        LEFT JOIN ml_training_dataset m ON p.ts_code = m.ts_code AND p.trade_date = m.trade_date
        WHERE p.trade_date >= $1 AND p.trade_date <= $2
          AND p.predicted_return IS NOT NULL
          AND p.confidence >= $3
        "#,
    );

    if cli.long_only {
        query.push_str(" AND p.predicted_return > 0");
    }

    query.push_str(
        " ORDER BY p.trade_date, (ABS(p.predicted_return) * 10 + p.confidence * 50) DESC",
    );

    let rows = sqlx::query(&query)
        .bind(start_date)
        .bind(end_date)
        .bind(cli.min_confidence)
        .fetch_all(pool)
        .await?;

    let signals: Vec<SignalWithReturn> = rows
        .iter()
        .filter(|row| {
            let signal_type: String = row.get(2);
            signal_type != "HOLD"
        })
        .map(|row| SignalWithReturn {
            ts_code: row.get(0),
            trade_date: row.get(1),
            signal_type: row.get(2),
            signal_score: row.get(3),
            confidence: row.get(4),
            predicted_return: row.get(5),
            actual_return_1day: row.get(6),
            actual_return_3day: row.get(7),
        })
        .collect();

    Ok(signals)
}

fn run_backtest(signals: &[SignalWithReturn], cli: &Cli) -> BacktestResult {
    let mut trades: Vec<Trade> = Vec::new();
    let mut daily_portfolio: Vec<DailyPortfolio> = Vec::new();

    let mut capital = cli.initial_capital;
    let mut cumulative_pnl = 0.0;

    let mut signals_by_date: HashMap<String, Vec<&SignalWithReturn>> = HashMap::new();
    for signal in signals {
        signals_by_date
            .entry(signal.trade_date.clone())
            .or_insert_with(Vec::new)
            .push(signal);
    }

    let mut dates: Vec<String> = signals_by_date.keys().cloned().collect();
    dates.sort();

    let mut peak_capital = cli.initial_capital;
    let mut max_drawdown = 0.0;

    for date in &dates {
        let day_signals = signals_by_date.get(date).unwrap();
        let selected: Vec<_> = day_signals.iter().take(cli.max_positions).collect();

        let mut daily_pnl = 0.0;
        let position_capital = capital * cli.position_size;

        for signal in &selected {
            let actual_return = if cli.use_3day {
                signal.actual_return_3day.unwrap_or(0.0)
            } else {
                signal.actual_return_1day.unwrap_or(0.0)
            };

            if actual_return.abs() < 1e-10 && signal.actual_return_1day.is_none() {
                continue;
            }

            let direction = match signal.signal_type.as_str() {
                "STRONG_BUY" | "BUY" => 1.0,
                "STRONG_SELL" | "SELL" => -1.0,
                _ => 0.0,
            };

            let gross_pnl = position_capital * direction * (actual_return / 100.0);
            let costs = position_capital * cli.transaction_cost * 2.0;
            let net_pnl = gross_pnl - costs;

            daily_pnl += net_pnl;

            let is_winner = net_pnl > 0.0;
            trades.push(Trade {
                ts_code: signal.ts_code.clone(),
                entry_date: signal.trade_date.clone(),
                signal_type: signal.signal_type.clone(),
                signal_score: signal.signal_score,
                confidence: signal.confidence,
                predicted_return: signal.predicted_return,
                actual_return,
                pnl: net_pnl,
                is_winner,
            });
        }

        capital += daily_pnl;
        cumulative_pnl += daily_pnl;

        if capital > peak_capital {
            peak_capital = capital;
        }
        let drawdown = (peak_capital - capital) / peak_capital;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }

        daily_portfolio.push(DailyPortfolio {
            date: date.clone(),
            capital,
            positions: selected.len(),
            cumulative_return: (capital - cli.initial_capital) / cli.initial_capital * 100.0,
        });
    }

    let total_trades = trades.len();
    let winning_trades = trades.iter().filter(|t| t.is_winner).count();
    let losing_trades = total_trades - winning_trades;

    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };

    let total_wins: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let total_losses: f64 = trades
        .iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| t.pnl.abs())
        .sum();

    let avg_win = if winning_trades > 0 {
        total_wins / winning_trades as f64
    } else {
        0.0
    };
    let avg_loss = if losing_trades > 0 {
        total_losses / losing_trades as f64
    } else {
        0.0
    };

    let profit_factor = if total_losses > 0.0 {
        total_wins / total_losses
    } else if total_wins > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let total_return = (capital - cli.initial_capital) / cli.initial_capital * 100.0;

    let trading_days = dates.len().max(1) as f64;
    let years = trading_days / 250.0;
    let annualized_return = if years > 0.0 {
        ((1.0 + total_return / 100.0).powf(1.0 / years) - 1.0) * 100.0
    } else {
        total_return
    };

    let daily_returns: Vec<f64> = daily_portfolio
        .iter()
        .map(|d| (d.capital - cli.initial_capital) / cli.initial_capital)
        .collect();

    let avg_daily_return = if !daily_returns.is_empty() {
        daily_returns.iter().sum::<f64>() / daily_returns.len() as f64
    } else {
        0.0
    };

    let daily_volatility = if daily_returns.len() > 1 {
        let variance: f64 = daily_returns
            .iter()
            .map(|r| (r - avg_daily_return).powi(2))
            .sum::<f64>()
            / (daily_returns.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let sharpe_ratio = if daily_volatility > 0.0 {
        (avg_daily_return * 250.0_f64.sqrt()) / daily_volatility
    } else {
        0.0
    };

    BacktestResult {
        trades,
        daily_portfolio,
        total_return,
        annualized_return,
        sharpe_ratio,
        max_drawdown: max_drawdown * 100.0,
        win_rate,
        avg_win,
        avg_loss,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
    }
}

fn display_results(result: &BacktestResult, cli: &Cli) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                   Backtest Results                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let final_capital = result
        .daily_portfolio
        .last()
        .map(|d| d.capital)
        .unwrap_or(cli.initial_capital);

    println!("┌───────────────────────────────────────────────────────────┐");
    println!("│  📊 Performance Summary                                   │");
    println!("├───────────────────────────────────────────────────────────┤");
    println!(
        "│  Initial Capital:     {:>20}             │",
        format_money(cli.initial_capital)
    );
    println!(
        "│  Final Capital:       {:>20}             │",
        format_money(final_capital)
    );
    println!(
        "│  Net Profit/Loss:     {:>20}             │",
        format_money_signed(final_capital - cli.initial_capital)
    );
    println!("│                                                           │");
    println!(
        "│  Total Return:        {:>+18.2}%             │",
        result.total_return
    );
    println!(
        "│  Annualized Return:   {:>+18.2}%             │",
        result.annualized_return
    );
    println!(
        "│  Sharpe Ratio:        {:>20.2}             │",
        result.sharpe_ratio
    );
    println!(
        "│  Max Drawdown:        {:>18.2}%             │",
        result.max_drawdown
    );
    println!("└───────────────────────────────────────────────────────────┘");
    println!();

    println!("┌───────────────────────────────────────────────────────────┐");
    println!("│  📈 Trade Statistics                                      │");
    println!("├───────────────────────────────────────────────────────────┤");
    println!(
        "│  Total Trades:        {:>20}             │",
        result.total_trades
    );
    println!(
        "│  Winning Trades:      {:>20}             │",
        result.winning_trades
    );
    println!(
        "│  Losing Trades:       {:>20}             │",
        result.losing_trades
    );
    println!("│                                                           │");
    println!(
        "│  Win Rate:            {:>18.1}%             │",
        result.win_rate
    );
    println!(
        "│  Avg Win:             {:>20}             │",
        format_money(result.avg_win)
    );
    println!(
        "│  Avg Loss:            {:>20}             │",
        format_money(result.avg_loss)
    );
    println!(
        "│  Profit Factor:       {:>20.2}             │",
        result.profit_factor
    );
    println!("└───────────────────────────────────────────────────────────┘");
    println!();

    // Signal type breakdown
    let mut buy_trades = 0;
    let mut sell_trades = 0;
    let mut buy_pnl = 0.0;
    let mut sell_pnl = 0.0;
    let mut buy_wins = 0;
    let mut sell_wins = 0;

    for trade in &result.trades {
        match trade.signal_type.as_str() {
            "STRONG_BUY" | "BUY" => {
                buy_trades += 1;
                buy_pnl += trade.pnl;
                if trade.is_winner {
                    buy_wins += 1;
                }
            }
            "STRONG_SELL" | "SELL" => {
                sell_trades += 1;
                sell_pnl += trade.pnl;
                if trade.is_winner {
                    sell_wins += 1;
                }
            }
            _ => {}
        }
    }

    println!("┌───────────────────────────────────────────────────────────┐");
    println!("│  🎯 Signal Type Breakdown                                 │");
    println!("├───────────────────────────────────────────────────────────┤");
    if buy_trades > 0 {
        let buy_win_rate = buy_wins as f64 / buy_trades as f64 * 100.0;
        println!(
            "│  BUY Signals:  {:>5} trades, {:>18} P&L      │",
            buy_trades,
            format_money_signed(buy_pnl)
        );
        println!(
            "│                Win Rate: {:>6.1}%                        │",
            buy_win_rate
        );
    }
    if sell_trades > 0 {
        let sell_win_rate = sell_wins as f64 / sell_trades as f64 * 100.0;
        println!(
            "│  SELL Signals: {:>5} trades, {:>18} P&L      │",
            sell_trades,
            format_money_signed(sell_pnl)
        );
        println!(
            "│                Win Rate: {:>6.1}%                        │",
            sell_win_rate
        );
    }
    println!("└───────────────────────────────────────────────────────────┘");
    println!();

    // Prediction accuracy
    let mut direction_correct = 0;
    let mut direction_total = 0;

    for trade in &result.trades {
        if trade.actual_return.abs() > 0.001 {
            direction_total += 1;
            let predicted_up = trade.predicted_return > 0.0;
            let actual_up = trade.actual_return > 0.0;
            if predicted_up == actual_up {
                direction_correct += 1;
            }
        }
    }

    if direction_total > 0 {
        let direction_accuracy = direction_correct as f64 / direction_total as f64 * 100.0;
        println!("┌───────────────────────────────────────────────────────────┐");
        println!("│  🎲 Prediction Accuracy                                   │");
        println!("├───────────────────────────────────────────────────────────┤");
        println!(
            "│  Direction Accuracy:  {:>18.1}%             │",
            direction_accuracy
        );
        println!(
            "│  (Correct: {:>5} / {:>5})                               │",
            direction_correct, direction_total
        );
        println!("└───────────────────────────────────────────────────────────┘");
        println!();
    }

    // Best and worst trades
    if !result.trades.is_empty() {
        let mut sorted_trades = result.trades.clone();
        sorted_trades.sort_by(|a, b| {
            b.pnl
                .partial_cmp(&a.pnl)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!("┌───────────────────────────────────────────────────────────┐");
        println!("│  🏆 Top 5 Best Trades                                     │");
        println!("├──────────┬──────────┬──────────┬──────────┬──────────────┤");
        println!("│ Code     │ Date     │ Signal   │ Return   │ P&L          │");
        println!("├──────────┼──────────┼──────────┼──────────┼──────────────┤");
        for trade in sorted_trades.iter().take(5) {
            println!(
                "│ {:8} │ {:8} │ {:8} │ {:+7.2}% │ {:>12} │",
                trade.ts_code,
                &trade.entry_date[..8],
                trade.signal_type,
                trade.actual_return,
                format_pnl(trade.pnl)
            );
        }
        println!("└──────────┴──────────┴──────────┴──────────┴──────────────┘");
        println!();

        println!("┌───────────────────────────────────────────────────────────┐");
        println!("│  💀 Top 5 Worst Trades                                    │");
        println!("├──────────┬──────────┬──────────┬──────────┬──────────────┤");
        println!("│ Code     │ Date     │ Signal   │ Return   │ P&L          │");
        println!("├──────────┼──────────┼──────────┼──────────┼──────────────┤");
        for trade in sorted_trades.iter().rev().take(5) {
            println!(
                "│ {:8} │ {:8} │ {:8} │ {:+7.2}% │ {:>12} │",
                trade.ts_code,
                &trade.entry_date[..8],
                trade.signal_type,
                trade.actual_return,
                format_pnl(trade.pnl)
            );
        }
        println!("└──────────┴──────────┴──────────┴──────────┴──────────────┘");
        println!();
    }

    // Equity curve
    if !result.daily_portfolio.is_empty() {
        println!("┌───────────────────────────────────────────────────────────┐");
        println!("│  📉 Equity Curve (Last 10 Days)                           │");
        println!("├──────────┬─────────────────────┬───────────┬─────────────┤");
        println!("│ Date     │ Capital             │ Positions │ Cum. Return │");
        println!("├──────────┼─────────────────────┼───────────┼─────────────┤");
        for day in result.daily_portfolio.iter().rev().take(10).rev() {
            println!(
                "│ {:8} │ {:>19} │ {:>9} │ {:>+10.2}% │",
                &day.date[..8],
                format_money(day.capital),
                day.positions,
                day.cumulative_return
            );
        }
        println!("└──────────┴─────────────────────┴───────────┴─────────────┘");
    }

    println!();

    // Risk assessment
    let risk_rating = if result.sharpe_ratio >= 2.0 && result.max_drawdown < 10.0 {
        "⭐⭐⭐⭐⭐ EXCELLENT"
    } else if result.sharpe_ratio >= 1.5 && result.max_drawdown < 15.0 {
        "⭐⭐⭐⭐ GOOD"
    } else if result.sharpe_ratio >= 1.0 && result.max_drawdown < 20.0 {
        "⭐⭐⭐ MODERATE"
    } else if result.sharpe_ratio >= 0.5 {
        "⭐⭐ FAIR"
    } else {
        "⭐ POOR"
    };

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Risk Assessment: {:38} ║", risk_rating);
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
}

async fn update_signal_actuals(pool: &Pool<Postgres>, trades: &[Trade]) -> Result<()> {
    let mut updated = 0;

    for trade in trades {
        let result = sqlx::query(
            r#"
            UPDATE trading_signals
            SET actual_return_1day = $3,
                signal_correct_1day = CASE 
                    WHEN signal_type IN ('STRONG_BUY', 'BUY') AND $3 > 0 THEN true
                    WHEN signal_type IN ('STRONG_SELL', 'SELL') AND $3 < 0 THEN true
                    ELSE false
                END
            WHERE ts_code = $1 AND trade_date = $2
              AND actual_return_1day IS NULL
            "#,
        )
        .bind(&trade.ts_code)
        .bind(&trade.entry_date)
        .bind(trade.actual_return)
        .execute(pool)
        .await?;

        if result.rows_affected() > 0 {
            updated += 1;
        }
    }

    if updated > 0 {
        println!("✓ Updated {} signals with actual return data", updated);
    }

    Ok(())
}
