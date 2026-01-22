/// Daily feature calculator: Computes ML features for all stocks for a given trading day
///
/// This binary runs incrementally (don't recalculate entire history):
/// - Fetches yesterday's market data
/// - Calculates technical indicators for yesterday's close
/// - Outputs normalized feature vectors for model inference
///
/// Usage:
///   cargo run --release --bin daily_features -- --date 2026-01-15
///   cargo run --release --bin daily_features -- --latest  # Uses yesterday
use chrono::NaiveDate;
use clap::Parser;
use rust_llm_stock::{
    db::MlTrainingRecord, feature_normalization::normalize_features, stock_db::get_connection,
};
use std::error::Error;

#[derive(Parser, Debug)]
#[command(name = "daily_features")]
#[command(about = "Calculate ML features for daily stock data")]
struct Args {
    /// Trade date in YYYYMMDD format (default: yesterday)
    #[arg(long)]
    date: Option<String>,

    /// Use yesterday's date automatically
    #[arg(long, default_value_t = true)]
    latest: bool,

    /// Output file path (default: stdout as CSV)
    #[arg(long)]
    output: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let args = Args::parse();

    // Determine trade date
    let trade_date = if let Some(d) = args.date {
        d
    } else {
        // Calculate yesterday
        let today = chrono::Local::now().naive_local().date();
        let yesterday = today - chrono::Duration::days(1);
        yesterday.format("%Y%m%d").to_string()
    };

    println!("=== Daily Feature Calculator ===");
    println!("Trade date: {}", trade_date);

    // Validate date format
    let _parsed_date = NaiveDate::parse_from_str(&trade_date, "%Y%m%d")
        .map_err(|e| format!("Invalid date format (YYYYMMDD): {}", e))?;

    // Connect to database
    let pool = get_connection().await;

    // Fetch all stocks with data for the target date
    println!("\nFetching stocks for {}...", trade_date);
    let stocks: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT ts_code FROM ml_training_dataset WHERE trade_date <= $1 ORDER BY ts_code",
    )
    .bind(&trade_date)
    .fetch_all(&pool)
    .await?;

    println!("Found {} stocks", stocks.len());

    let mut feature_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    let mut success_count = 0;
    let mut skip_count = 0;

    // Calculate features for each stock
    for (idx, ts_code) in stocks.iter().enumerate() {
        // Debug first few stocks
        if idx < 3 {
            println!("  🔍 Querying stock {} for date {}", ts_code, trade_date);
        }

        let record_opt: Option<MlTrainingRecord> = sqlx::query_as::<_, MlTrainingRecord>(
            r#"
            SELECT 
                ts_code, trade_date, industry, act_ent_type, volume, amount,
                month, weekday, quarter, week_no as weekno,
                open_pct, high_pct, low_pct, close_pct,
                high_from_open_pct, low_from_open_pct, close_from_open_pct, intraday_range_pct, close_position_in_range,
                ema_5, ema_10, ema_20, ema_30, ema_60, sma_5, sma_10, sma_20,
                macd_line, macd_signal, macd_histogram, macd_weekly_line, macd_weekly_signal, macd_monthly_line, macd_monthly_signal,
                rsi_14, kdj_k, kdj_d, kdj_j,
                bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b,
                atr, volatility_5, volatility_20, asi, obv, volume_ratio,
                price_momentum_5, price_momentum_10, price_momentum_20, price_position_52w,
                body_size, upper_shadow, lower_shadow,
                trend_strength, adx_14, vwap_distance_pct, cmf_20, williams_r_14,
                aroon_up_25, aroon_down_25,
                return_lag_1, return_lag_2, return_lag_3,
                overnight_gap, gap_pct,
                volume_roc_5, volume_spike,
                price_roc_5, price_roc_10, price_roc_20,
                hist_volatility_20,
                is_doji, is_hammer, is_shooting_star, consecutive_days,
                index_csi300_pct_chg, index_csi300_vs_ma5_pct, index_csi300_vs_ma20_pct,
                index_xin9_pct_chg, index_xin9_vs_ma5_pct, index_xin9_vs_ma20_pct,
                index_chinext_pct_chg, index_chinext_vs_ma5_pct, index_chinext_vs_ma20_pct,
                vol_percentile, high_vol_regime,
                NULL::DOUBLE PRECISION as industry_avg_return,
                NULL::DOUBLE PRECISION as stock_vs_industry,
                NULL::DOUBLE PRECISION as industry_momentum_5d,
                next_day_return, next_day_direction,
                turnover_rate, turnover_rate_f, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                total_share, float_share, free_share, total_mv, circ_mv,
                pe_percentile_52w, sector_momentum_vs_market, volume_accel_5d, price_vs_52w_high, consecutive_up_days,
                index_hsi_pct_chg, index_hsi_vs_ma5_pct, index_hsi_vs_ma20_pct,
                fx_usdcnh_pct_chg, fx_usdcnh_vs_ma5_pct, fx_usdcnh_vs_ma20_pct,
                net_mf_vol, net_mf_amount, smart_money_ratio, large_order_flow
            FROM ml_training_dataset
            WHERE ts_code = $1 AND trade_date <= $2
            ORDER BY trade_date DESC
            LIMIT 1
            "#
        )
        .bind(ts_code)
        .bind(&trade_date)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        if let Some(record) = record_opt {
            // Debug first successful record
            if success_count == 0 {
                println!(
                    "  ✅ First record found: {} on {}",
                    record.ts_code, record.trade_date
                );
            }

            let features = extract_features_from_record(&record);
            let reference_close_pct = record.close_pct.unwrap_or(0.0).abs().max(0.01);
            let normalized = normalize_features(features, reference_close_pct as f32);

            feature_vectors.push((ts_code.clone(), normalized.to_vec()));
            success_count += 1;
            if (idx + 1) % 50 == 0 {
                println!("  [{}/{}] {}: OK", idx + 1, stocks.len(), ts_code);
            }
        } else {
            skip_count += 1;
            if skip_count <= 5 {
                println!("  ⚠️  {}: No data for {} (skipped)", ts_code, trade_date);
            }
        }
    }

    println!("\n✅ Calculated features for {} stocks", success_count);
    if skip_count > 0 {
        println!("⚠️  Skipped {} stocks (no data)", skip_count);
    }

    // Output as CSV
    if let Some(output_path) = args.output {
        let mut wtr = csv::Writer::from_path(&output_path)?;

        // Write header
        let mut header = vec!["ts_code".to_string()];
        for i in 0..105 {
            header.push(format!("feature_{:03}", i));
        }
        wtr.write_record(&header)?;

        // Write rows
        for (ts_code, features) in &feature_vectors {
            let mut row = vec![ts_code.clone()];
            for f in features {
                row.push(f.to_string());
            }
            wtr.write_record(&row)?;
        }

        wtr.flush()?;
        println!("\n📄 Features written to: {}", output_path);
    } else {
        // Write to stdout as simple CSV
        println!("\nts_code,feature_000,feature_001,...,feature_104");
        for (ts_code, features) in &feature_vectors {
            print!("{}", ts_code);
            for f in features {
                print!(",{}", f);
            }
            println!();
        }
    }

    Ok(())
}

/// Extract 105 features from a database record
/// Follows the layout from src/dataset.rs StockItem
fn extract_features_from_record(record: &MlTrainingRecord) -> [f32; 105] {
    [
        // [0-2] Categorical (placeholder - would be label-encoded)
        0.0,
        0.0,
        0.0,
        // [3-4] Volume
        record.volume as f32,
        record.amount.unwrap_or(0.0) as f32,
        // [5-8] Temporal
        record.month.unwrap_or(1) as f32,
        record.weekday.unwrap_or(0) as f32,
        record.quarter.unwrap_or(1) as f32,
        record.weekno.unwrap_or(0) as f32,
        // [9-12] Price percentages
        record.open_pct.unwrap_or(0.0) as f32,
        record.high_pct.unwrap_or(0.0) as f32,
        record.low_pct.unwrap_or(0.0) as f32,
        record.close_pct.unwrap_or(0.0) as f32,
        // [13-17] Intraday movements
        record.high_from_open_pct.unwrap_or(0.0) as f32,
        record.low_from_open_pct.unwrap_or(0.0) as f32,
        record.close_from_open_pct.unwrap_or(0.0) as f32,
        record.intraday_range_pct.unwrap_or(0.0) as f32,
        record.close_position_in_range.unwrap_or(0.5) as f32,
        // [18-25] Moving averages
        record.ema_5.unwrap_or(0.0) as f32,
        record.ema_10.unwrap_or(0.0) as f32,
        record.ema_20.unwrap_or(0.0) as f32,
        record.ema_30.unwrap_or(0.0) as f32,
        record.ema_60.unwrap_or(0.0) as f32,
        record.sma_5.unwrap_or(0.0) as f32,
        record.sma_10.unwrap_or(0.0) as f32,
        record.sma_20.unwrap_or(0.0) as f32,
        // [26-32] MACD
        record.macd_line.unwrap_or(0.0) as f32,
        record.macd_signal.unwrap_or(0.0) as f32,
        record.macd_histogram.unwrap_or(0.0) as f32,
        record.macd_weekly_line.unwrap_or(0.0) as f32,
        record.macd_weekly_signal.unwrap_or(0.0) as f32,
        record.macd_monthly_line.unwrap_or(0.0) as f32,
        record.macd_monthly_signal.unwrap_or(0.0) as f32,
        // [33-36] Technical indicators
        record.rsi_14.unwrap_or(50.0) as f32,
        record.kdj_k.unwrap_or(50.0) as f32,
        record.kdj_d.unwrap_or(50.0) as f32,
        record.kdj_j.unwrap_or(50.0) as f32,
        // [37-41] Bollinger Bands
        record.bb_upper.unwrap_or(0.0) as f32,
        record.bb_middle.unwrap_or(0.0) as f32,
        record.bb_lower.unwrap_or(0.0) as f32,
        record.bb_bandwidth.unwrap_or(0.0) as f32,
        record.bb_percent_b.unwrap_or(0.5) as f32,
        // [42-47] Volatility
        record.atr.unwrap_or(0.0) as f32,
        record.volatility_5.unwrap_or(0.0) as f32,
        record.volatility_20.unwrap_or(0.0) as f32,
        record.asi.unwrap_or(0.0) as f32,
        record.obv.unwrap_or(0.0) as f32,
        record.volume_ratio.unwrap_or(1.0) as f32,
        // [48-51] Momentum
        record.price_momentum_5.unwrap_or(0.0) as f32,
        record.price_momentum_10.unwrap_or(0.0) as f32,
        record.price_momentum_20.unwrap_or(0.0) as f32,
        record.price_position_52w.unwrap_or(0.5) as f32,
        // [52-54] Candlestick sizes
        record.body_size.unwrap_or(0.0) as f32,
        record.upper_shadow.unwrap_or(0.0) as f32,
        record.lower_shadow.unwrap_or(0.0) as f32,
        // [55-62] Trend & strength
        record.trend_strength.unwrap_or(0.0) as f32,
        record.adx_14.unwrap_or(25.0) as f32,
        record.vwap_distance_pct.unwrap_or(0.0) as f32,
        record.cmf_20.unwrap_or(0.0) as f32,
        50.0 as f32,
        record.williams_r_14.unwrap_or(-50.0) as f32,
        record.aroon_up_25.unwrap_or(50.0) as f32,
        record.aroon_down_25.unwrap_or(50.0) as f32,
        // [63-65] Lagged returns
        record.return_lag_1.unwrap_or(0.0) as f32,
        record.return_lag_2.unwrap_or(0.0) as f32,
        record.return_lag_3.unwrap_or(0.0) as f32,
        // [66-67] Gap features
        record.overnight_gap.unwrap_or(0.0) as f32,
        record.gap_pct.unwrap_or(0.0) as f32,
        // [68-69] Volume features
        record.volume_roc_5.unwrap_or(0.0) as f32,
        if record.volume_spike.unwrap_or(false) {
            1.0
        } else {
            0.0
        },
        // [70-73] Price ROC & volatility
        record.price_roc_5.unwrap_or(0.0) as f32,
        record.price_roc_10.unwrap_or(0.0) as f32,
        record.price_roc_20.unwrap_or(0.0) as f32,
        record.hist_volatility_20.unwrap_or(0.0) as f32,
        // [74-77] Candlestick patterns
        if record.is_doji.unwrap_or(false) {
            1.0
        } else {
            0.0
        },
        if record.is_hammer.unwrap_or(false) {
            1.0
        } else {
            0.0
        },
        if record.is_shooting_star.unwrap_or(false) {
            1.0
        } else {
            0.0
        },
        record.consecutive_days.unwrap_or(0) as f32,
        // [78-86] Index features (CSI300, XIN9, ChiNext)
        record.index_csi300_pct_chg.unwrap_or(0.0) as f32,
        record.index_csi300_vs_ma5_pct.unwrap_or(0.0) as f32,
        record.index_csi300_vs_ma20_pct.unwrap_or(0.0) as f32,
        record.index_xin9_pct_chg.unwrap_or(0.0) as f32,
        record.index_xin9_vs_ma5_pct.unwrap_or(0.0) as f32,
        record.index_xin9_vs_ma20_pct.unwrap_or(0.0) as f32,
        record.index_chinext_pct_chg.unwrap_or(0.0) as f32,
        record.index_chinext_vs_ma5_pct.unwrap_or(0.0) as f32,
        record.index_chinext_vs_ma20_pct.unwrap_or(0.0) as f32,
        // [87-92] HSI & FX (placeholder if not in DB)
        record.index_hsi_pct_chg.unwrap_or(0.0) as f32,
        record.index_hsi_vs_ma5_pct.unwrap_or(0.0) as f32,
        record.index_hsi_vs_ma20_pct.unwrap_or(0.0) as f32,
        record.fx_usdcnh_pct_chg.unwrap_or(0.0) as f32,
        record.fx_usdcnh_vs_ma5_pct.unwrap_or(0.0) as f32,
        record.fx_usdcnh_vs_ma20_pct.unwrap_or(0.0) as f32,
        // [93-96] Money flow
        record.net_mf_vol.unwrap_or(0.0) as f32,
        record.net_mf_amount.unwrap_or(0.0) as f32,
        record.smart_money_ratio.unwrap_or(0.0) as f32,
        record.large_order_flow.unwrap_or(0.0) as f32,
        // [97-99] Industry features
        record.industry_avg_return.unwrap_or(0.0) as f32,
        record.stock_vs_industry.unwrap_or(0.0) as f32,
        record.industry_momentum_5d.unwrap_or(0.0) as f32,
        // [100-101] Volatility regime
        record.vol_percentile.unwrap_or(0.5) as f32,
        if record.high_vol_regime.unwrap_or(0) != 0 {
            1.0
        } else {
            0.0
        },
        // [102-104] Reserved for future use
        0.0,
        0.0,
        0.0,
    ]
}
