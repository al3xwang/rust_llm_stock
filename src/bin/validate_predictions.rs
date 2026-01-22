use anyhow::Result;
use clap::Parser;
use rust_llm_stock::stock_db::get_connection;
use sqlx::Row; // for row.get(..)

#[derive(Parser, Debug)]
#[command(author, version, about = "Validate stock predictions against actual market data", long_about = None)]
struct Args {
    /// Specific trade date to validate (YYYYMMDD). If omitted, validates all pending predictions
    #[arg(long, alias = "date", value_name = "YYYYMMDD")]
    validate_date: Option<String>,

    /// Only validate 1-day predictions
    #[arg(long)]
    only_1day: bool,

    /// Only validate 3-day predictions
    #[arg(long)]
    only_3day: bool,

    /// Show detailed statistics
    #[arg(short, long)]
    verbose: bool,

    /// Limit number of predictions to validate (for testing)
    #[arg(short, long)]
    limit: Option<usize>,

    /// Force revalidation of already-validated predictions (ignores actual_return IS NULL filter)
    #[arg(long)]
    force: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let pool = get_connection().await;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Stock Predictions Validation Tool               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Validate 1-day predictions
    if !args.only_3day {
        println!("=== Validating 1-Day Predictions ===\n");
        let stats_1day =
            validate_1day_predictions(&pool, args.validate_date.as_deref(), args.limit, args.force)
                .await?;
        print_statistics("1-Day", &stats_1day, args.verbose);
        println!();
    }

    // Validate 3-day predictions
    if !args.only_1day {
        println!("=== Validating 3-Day Predictions ===\n");
        let stats_3day =
            validate_3day_predictions(&pool, args.validate_date.as_deref(), args.limit, args.force)
                .await?;
        print_statistics("3-Day", &stats_3day, args.verbose);
        println!();
    }

    Ok(())
}

struct ValidationStats {
    total_validated: usize,
    correct_predictions: usize,
    incorrect_predictions: usize,
    accuracy_rate: f64,
    avg_predicted_return: f64,
    avg_actual_return: f64,
    avg_prediction_error: f64,
    high_confidence_correct: usize,
    high_confidence_total: usize,
    confidence_buckets: Vec<ConfidenceBucketStats>,
}

#[derive(Clone, Debug)]
struct ConfidenceBucketStats {
    label: String,
    correct: usize,
    total: usize,
}

/// Validate 1-day predictions by comparing with next day's actual data
async fn validate_1day_predictions(
    pool: &sqlx::Pool<sqlx::Postgres>,
    specific_date: Option<&str>,
    limit: Option<usize>,
    force: bool,
) -> Result<ValidationStats> {
    // Build query based on parameters
    let null_filter = if force {
        ""
    } else {
        "AND sp.actual_return IS NULL"
    };
    let query = if let Some(date) = specific_date {
        format!(
            "SELECT sp.ts_code, sp.trade_date, sp.predicted_direction, sp.predicted_return, sp.confidence
             FROM stock_predictions sp
             WHERE sp.trade_date = '{}'
               {}
             ORDER BY sp.trade_date DESC, sp.ts_code
             {}",
            date,
            null_filter,
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        )
    } else {
        format!(
            "SELECT sp.ts_code, sp.trade_date, sp.predicted_direction, sp.predicted_return, sp.confidence
             FROM stock_predictions sp
             WHERE {} 
               AND EXISTS (
                   -- Only validate if next day's data is available
                   SELECT 1 FROM adjusted_stock_daily next_day
                   WHERE next_day.ts_code = sp.ts_code
                     AND next_day.trade_date > sp.trade_date
               )
             ORDER BY sp.trade_date DESC, sp.ts_code
             {}",
            if force { "1=1" } else { "sp.actual_return IS NULL" },
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        )
    };

    let rows = sqlx::query(&query).fetch_all(pool).await?;

    println!("Found {} predictions to validate", rows.len());

    let mut validated = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    let mut sum_predicted = 0.0;
    let mut sum_actual = 0.0;
    let mut sum_error = 0.0;
    let mut high_conf_correct = 0;
    let mut high_conf_total = 0;
    let mut missing_next_day = 0;
    let mut confidence_buckets = vec![
        ConfidenceBucketStats {
            label: "0.0-0.1".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.1-0.2".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.2-0.3".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.3-0.4".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.4-0.5".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.5-0.6".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.6-0.7".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.7-0.8".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.8-0.9".to_string(),
            correct: 0,
            total: 0,
        },
        ConfidenceBucketStats {
            label: "0.9-1.0".to_string(),
            correct: 0,
            total: 0,
        },
    ];

    for row in rows {
        let ts_code: String = row.get(0);
        let trade_date: String = row.get(1);
        let predicted_direction: bool = row.get(2);
        let predicted_return: f64 = row.get(3);
        let confidence: f64 = row.get(4);

        // Get actual next-day data
        let actual_data = sqlx::query!(
            r#"
            SELECT 
                ((next_day.close - current_day.close) / current_day.close * 100.0)::float8 AS "actual_return?",
                (next_day.close > current_day.close) AS "actual_direction?"
            FROM adjusted_stock_daily current_day
            JOIN adjusted_stock_daily next_day 
                ON next_day.ts_code = current_day.ts_code
                AND next_day.trade_date = (
                    SELECT MIN(trade_date) 
                    FROM adjusted_stock_daily 
                    WHERE ts_code = current_day.ts_code 
                    AND trade_date > current_day.trade_date
                )
            WHERE current_day.ts_code = $1 
            AND current_day.trade_date = $2
            "#,
            ts_code,
            trade_date
        )
        .fetch_optional(pool)
        .await?;

        if let Some(actual) = actual_data {
            let actual_return = actual.actual_return.unwrap_or(0.0);
            let actual_direction = actual.actual_direction.unwrap_or(false);
            let is_correct = predicted_direction == actual_direction;

            // Update database
            sqlx::query!(
                r#"
                UPDATE stock_predictions
                SET actual_return = $1,
                    actual_direction = $2,
                    prediction_correct = $3
                WHERE ts_code = $4
                  AND trade_date = $5
                "#,
                actual_return,
                actual_direction,
                is_correct,
                ts_code,
                trade_date
            )
            .execute(pool)
            .await?;

            // Accumulate statistics
            validated += 1;
            if is_correct {
                correct += 1;
            } else {
                incorrect += 1;
            }

            sum_predicted += predicted_return;
            sum_actual += actual_return;
            sum_error += (predicted_return - actual_return).abs();

            // Track high confidence predictions (>0.7)
            if confidence > 0.7 {
                high_conf_total += 1;
                if is_correct {
                    high_conf_correct += 1;
                }
            }

            // Confidence buckets for segmented accuracy (10 buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
            let bucket_idx = ((confidence * 10.0).floor() as usize).min(9);
            confidence_buckets[bucket_idx].total += 1;
            if is_correct {
                confidence_buckets[bucket_idx].correct += 1;
            }

            if validated % 100 == 0 {
                print!("\rValidated: {} predictions...", validated);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        } else {
            // Could not find the next trading day; likely missing adjusted_stock_daily data
            missing_next_day += 1;
        }
    }

    println!("\rValidated: {} predictions    ", validated);
    if missing_next_day > 0 {
        println!(
            "  Skipped {} predictions (no next-day data in adjusted_stock_daily)",
            missing_next_day
        );
    }

    let accuracy_rate = if validated > 0 {
        (correct as f64 / validated as f64) * 100.0
    } else {
        0.0
    };

    let avg_predicted = if validated > 0 {
        sum_predicted / validated as f64
    } else {
        0.0
    };

    let avg_actual = if validated > 0 {
        sum_actual / validated as f64
    } else {
        0.0
    };

    let avg_error = if validated > 0 {
        sum_error / validated as f64
    } else {
        0.0
    };

    Ok(ValidationStats {
        total_validated: validated,
        correct_predictions: correct,
        incorrect_predictions: incorrect,
        accuracy_rate,
        avg_predicted_return: avg_predicted,
        avg_actual_return: avg_actual,
        avg_prediction_error: avg_error,
        high_confidence_correct: high_conf_correct,
        high_confidence_total: high_conf_total,
        confidence_buckets,
    })
}

/// Validate 3-day predictions by comparing with data 3 trading days later
async fn validate_3day_predictions(
    pool: &sqlx::Pool<sqlx::Postgres>,
    specific_date: Option<&str>,
    limit: Option<usize>,
    force: bool,
) -> Result<ValidationStats> {
    // Build query based on parameters
    let null_filter = if force {
        ""
    } else {
        "AND sp.actual_3day_return IS NULL"
    };
    let query = if let Some(date) = specific_date {
        format!(
            "SELECT sp.ts_code, sp.trade_date, sp.predicted_3day_direction, sp.predicted_3day_return
             FROM stock_predictions sp
             WHERE sp.trade_date = '{}'
               AND sp.predicted_3day_return IS NOT NULL
               {}
             ORDER BY sp.trade_date DESC, sp.ts_code
             {}",
            date,
            null_filter,
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        )
    } else {
        format!(
            "SELECT sp.ts_code, sp.trade_date, sp.predicted_3day_direction, sp.predicted_3day_return
             FROM stock_predictions sp
             WHERE sp.predicted_3day_return IS NOT NULL
               {}
               AND EXISTS (
                   -- Only validate if 3-day data is available
                   SELECT 1 FROM (
                       SELECT ts_code, trade_date,
                              ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date) as rn
                       FROM adjusted_stock_daily
                       WHERE ts_code = sp.ts_code
                         AND trade_date > sp.trade_date
                   ) future_days
                   WHERE future_days.rn >= 3
               )
             ORDER BY sp.trade_date DESC, sp.ts_code
             {}",
            null_filter,
            limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default()
        )
    };

    let rows = sqlx::query(&query).fetch_all(pool).await?;

    println!("Found {} 3-day predictions to validate", rows.len());

    let mut validated = 0;
    let mut correct = 0;
    let mut incorrect = 0;
    let mut sum_predicted = 0.0;
    let mut sum_actual = 0.0;
    let mut sum_error = 0.0;

    for row in rows {
        let ts_code: String = row.get(0);
        let trade_date: String = row.get(1);
        let predicted_direction: Option<bool> = row.get(2);
        let predicted_return: Option<f64> = row.get(3);

        if predicted_direction.is_none() || predicted_return.is_none() {
            continue;
        }

        let predicted_direction = predicted_direction.unwrap();
        let predicted_return = predicted_return.unwrap();

        // Get actual 3-day data (3rd trading day after prediction date)
        let actual_data = sqlx::query!(
            r#"
            SELECT 
                ((day3.close - current_day.close) / current_day.close * 100.0)::float8 AS "actual_return?",
                (day3.close > current_day.close) AS "actual_direction?"
            FROM adjusted_stock_daily current_day
            CROSS JOIN LATERAL (
                SELECT close, trade_date
                FROM adjusted_stock_daily
                WHERE ts_code = current_day.ts_code
                  AND trade_date > current_day.trade_date
                ORDER BY trade_date
                LIMIT 1 OFFSET 2
            ) day3
            WHERE current_day.ts_code = $1 
            AND current_day.trade_date = $2
            "#,
            ts_code,
            trade_date
        )
        .fetch_optional(pool)
        .await?;

        if let Some(actual) = actual_data {
            let actual_return = actual.actual_return.unwrap_or(0.0);
            let actual_direction = actual.actual_direction.unwrap_or(false);
            let is_correct = predicted_direction == actual_direction;

            // Update database
            sqlx::query!(
                r#"
                UPDATE stock_predictions
                SET actual_3day_return = $1,
                    actual_3day_direction = $2,
                    prediction_3day_correct = $3
                WHERE ts_code = $4
                  AND trade_date = $5
                "#,
                actual_return,
                actual_direction,
                is_correct,
                ts_code,
                trade_date
            )
            .execute(pool)
            .await?;

            // Accumulate statistics
            validated += 1;
            if is_correct {
                correct += 1;
            } else {
                incorrect += 1;
            }

            sum_predicted += predicted_return;
            sum_actual += actual_return;
            sum_error += (predicted_return - actual_return).abs();

            if validated % 100 == 0 {
                print!("\rValidated: {} predictions...", validated);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
    }

    println!("\rValidated: {} predictions    ", validated);

    let accuracy_rate = if validated > 0 {
        (correct as f64 / validated as f64) * 100.0
    } else {
        0.0
    };

    let avg_predicted = if validated > 0 {
        sum_predicted / validated as f64
    } else {
        0.0
    };

    let avg_actual = if validated > 0 {
        sum_actual / validated as f64
    } else {
        0.0
    };

    let avg_error = if validated > 0 {
        sum_error / validated as f64
    } else {
        0.0
    };

    Ok(ValidationStats {
        total_validated: validated,
        correct_predictions: correct,
        incorrect_predictions: incorrect,
        accuracy_rate,
        avg_predicted_return: avg_predicted,
        avg_actual_return: avg_actual,
        avg_prediction_error: avg_error,
        high_confidence_correct: 0,
        high_confidence_total: 0,
        confidence_buckets: Vec::new(),
    })
}

fn print_statistics(horizon: &str, stats: &ValidationStats, verbose: bool) {
    println!("┌─────────────────────────────────────────────────────┐");
    println!(
        "│  {} Prediction Results                         │",
        horizon
    );
    println!("└─────────────────────────────────────────────────────┘");
    println!();
    println!("  Total Validated:        {}", stats.total_validated);
    println!(
        "  Correct:                {} ({:.2}%)",
        stats.correct_predictions, stats.accuracy_rate
    );
    println!("  Incorrect:              {}", stats.incorrect_predictions);
    println!();
    println!(
        "  Avg Predicted Return:   {:.4}%",
        stats.avg_predicted_return
    );
    println!("  Avg Actual Return:      {:.4}%", stats.avg_actual_return);
    println!(
        "  Avg Prediction Error:   {:.4}%",
        stats.avg_prediction_error
    );

    if stats.high_confidence_total > 0 {
        let high_conf_accuracy =
            (stats.high_confidence_correct as f64 / stats.high_confidence_total as f64) * 100.0;
        println!();
        println!(
            "  High Confidence (>0.7): {} / {} ({:.2}%)",
            stats.high_confidence_correct, stats.high_confidence_total, high_conf_accuracy
        );
    }

    if !stats.confidence_buckets.is_empty() {
        println!();
        println!("  Confidence Buckets:");
        for bucket in &stats.confidence_buckets {
            let acc = if bucket.total > 0 {
                (bucket.correct as f64 / bucket.total as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "    {:>8}: {:>4} / {:<4} ({:.2}%)",
                bucket.label, bucket.correct, bucket.total, acc
            );
        }
    }

    if verbose && stats.total_validated > 0 {
        println!();
        println!(
            "  Prediction Bias:        {:.4}%",
            stats.avg_predicted_return - stats.avg_actual_return
        );
    }
}
