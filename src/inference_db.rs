use crate::dataset::StockItem;
use crate::db::{DbClient, MlTrainingRecord};
use crate::model::{ModelConfig, StockModel};
use anyhow::Result;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Predictor for stock price forecasting using trained model
pub struct StockPredictor<B: Backend> {
    model: StockModel<B>,
    device: B::Device,
    seq_len: usize,
}

impl<B: Backend> StockPredictor<B> {
    /// Load trained model from artifacts
    pub fn load(model_path: &str, device: B::Device) -> Result<Self> {
        println!("Loading model from {}...", model_path);

        let config = ModelConfig::new();
        let record = CompactRecorder::new()
            .load(model_path.into(), &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

        let model = config.init::<B>(&device).load_record(record);

        Ok(Self {
            model,
            device,
            seq_len: 30, // Must match training seq_len
        })
    }

    /// Predict next day's OHLCV + features from historical data
    pub fn predict(&self, stock_item: &StockItem) -> Result<Vec<f32>> {
        if stock_item.values.len() != self.seq_len {
            return Err(anyhow::anyhow!(
                "Stock item length {} doesn't match required seq_len {}",
                stock_item.values.len(),
                self.seq_len
            ));
        }

        // Normalize input features (same as training batcher)
        let mut normalized_data = Vec::with_capacity(self.seq_len * 19);
        for features in &stock_item.values {
            let mut feat = *features;
            // Normalize OHLC by close
            let close = feat[3].max(1.0);
            feat[0] /= close; // open
            feat[1] /= close; // high
            feat[2] /= close; // low
            feat[3] = 1.0; // close

            // Volume: log scale
            feat[4] = (feat[4].max(1.0)).ln() / 20.0;

            // Weekday: 0-1 range
            feat[5] /= 7.0;

            // Week_No: 0-1 range
            feat[6] /= 53.0;

            // EMA5/10/20/30/60: normalize by close
            feat[7] /= close;
            feat[8] /= close;
            feat[9] /= close;
            feat[10] /= close;
            feat[11] /= close;

            // Volume_ratio: clamp
            feat[12] = feat[12].clamp(0.0, 10.0);

            // MACD: clamp
            feat[13] = feat[13].clamp(-1.0, 1.0);
            feat[14] = feat[14].clamp(-1.0, 1.0);

            // ASI: clamp and normalize
            feat[15] = feat[15].clamp(-2000.0, 12000.0) / 1000.0;

            // OBV: log-normalize with sign preservation
            feat[16] = (feat[16].abs().max(1.0)).ln() / 18.0 * feat[16].signum();

            // Amount: log-normalize
            feat[17] = (feat[17].max(1.0)).ln() / 18.0;

            // pct_change: normalize to ±1 range
            feat[18] = feat[18].clamp(-10.0, 10.0) / 10.0;

            normalized_data.extend_from_slice(&feat);
        }

        // Create tensor [1, seq_len, 19]
        let input_data = TensorData::new(normalized_data, vec![1, self.seq_len, 19]);
        let input = Tensor::<B, 3>::from_data(input_data, &self.device);

        // Forward pass
        let output = self.model.forward(input);

        // Get last timestep prediction [1, seq_len, 19] -> take [0, seq_len-1, :]
        let output_data = output.into_data();
        let predictions = output_data.as_slice::<f32>().unwrap();

        // Extract last prediction (19 features)
        let last_idx = (self.seq_len - 1) * 19;
        let predicted_features: Vec<f32> = predictions[last_idx..last_idx + 19].to_vec();

        // Denormalize predictions back to original scale
        let last_close = stock_item.values.last().unwrap()[3];
        let mut denormalized = predicted_features.clone();

        denormalized[0] *= last_close; // open
        denormalized[1] *= last_close; // high
        denormalized[2] *= last_close; // low
        denormalized[3] *= last_close; // close
        denormalized[4] = (denormalized[4] * 20.0).exp(); // volume
        denormalized[5] *= 7.0; // weekday
        denormalized[6] *= 53.0; // week_no
        denormalized[7] *= last_close; // EMA5
        denormalized[8] *= last_close; // EMA10
        denormalized[9] *= last_close; // EMA20
        denormalized[10] *= last_close; // EMA30
        denormalized[11] *= last_close; // EMA60
        // denormalized[12-14] are already in reasonable ranges (volume_ratio, MACD)
        denormalized[15] *= 1000.0; // ASI
        // OBV: handle sign-preserving denormalization
        let obv_sign = if denormalized[16] >= 0.0 { 1.0 } else { -1.0 };
        denormalized[16] = (denormalized[16].abs() * 18.0).exp() * obv_sign; // OBV
        denormalized[17] = (denormalized[17] * 18.0).exp(); // Amount
        denormalized[18] *= 10.0; // pct_change

        Ok(denormalized)
    }

    /// Batch predict for multiple stock sequences
    pub fn predict_batch(&self, items: &[StockItem]) -> Result<Vec<Vec<f32>>> {
        items.iter().map(|item| self.predict(item)).collect()
    }
}

/// Test model accuracy on database records
pub async fn test_model_accuracy<B: Backend>(
    predictor: &StockPredictor<B>,
    db_client: &DbClient,
    ts_code: &str,
    start_date: &str,
    end_date: &str,
) -> Result<TestMetrics> {
    println!(
        "Testing model on {} from {} to {}",
        ts_code, start_date, end_date
    );

    // Fetch test data
    let records = db_client
        .fetch_stock_data(ts_code, start_date, end_date)
        .await?;

    println!(
        "Fetched {} records, need at least {} (seq_len={} + 1)",
        records.len(),
        predictor.seq_len + 1,
        predictor.seq_len
    );

    if records.len() < predictor.seq_len + 1 {
        return Err(anyhow::anyhow!("Not enough data for testing"));
    }

    let mut mse_sum = 0.0;
    let mut mae_sum = 0.0;
    let mut direction_correct = 0;
    let mut total_predictions = 0;

    // Create sliding windows
    for i in 0..records.len() - predictor.seq_len {
        let window = &records[i..i + predictor.seq_len];
        let actual_next = &records[i + predictor.seq_len];

        // Convert to StockItem
        let stock_item = db_records_to_stock_item(window);

        // Predict
        match predictor.predict(&stock_item) {
            Ok(predicted) => {
                // Focus on close price prediction (index 3)
                let predicted_close = predicted[3];
                let actual_close = actual_next.close as f32;
                let last_close = window.last().unwrap().close as f32;

                // Calculate metrics
                let error = predicted_close - actual_close;
                mse_sum += error * error;
                mae_sum += error.abs();

                // Direction accuracy
                let predicted_direction = predicted_close > last_close;
                let actual_direction = actual_close > last_close;
                if predicted_direction == actual_direction {
                    direction_correct += 1;
                }

                total_predictions += 1;

                if total_predictions % 50 == 0 {
                    println!("  Processed {} predictions...", total_predictions);
                }
            }
            Err(e) => {
                eprintln!("Prediction error: {}", e);
            }
        }
    }

    let metrics = TestMetrics {
        mse: mse_sum / total_predictions as f32,
        mae: mae_sum / total_predictions as f32,
        rmse: (mse_sum / total_predictions as f32).sqrt(),
        direction_accuracy: direction_correct as f32 / total_predictions as f32,
        total_predictions,
    };

    println!("\n=== Test Results ===");
    println!("Total Predictions: {}", metrics.total_predictions);
    println!("MSE:  {:.6}", metrics.mse);
    println!("MAE:  {:.6}", metrics.mae);
    println!("RMSE: {:.6}", metrics.rmse);
    println!(
        "Direction Accuracy: {:.2}%",
        metrics.direction_accuracy * 100.0
    );

    Ok(metrics)
}

#[derive(Debug, Clone)]
pub struct TestMetrics {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub direction_accuracy: f32,
    pub total_predictions: usize,
}

/// Convert database records to StockItem
pub fn db_records_to_stock_item(records: &[MlTrainingRecord]) -> StockItem {
    let values: Vec<[f32; 105]> = records
        .iter()
        .map(|r| {
            [
                // [0-2] Categorical (placeholder)
                0.0,
                0.0,
                0.0,
                // [3-4] Volume/Amount
                r.volume as f32,
                r.amount.unwrap_or(0.0) as f32,
                // [5-8] Temporal
                r.month.unwrap_or(1) as f32,
                r.weekday.unwrap_or(0) as f32,
                r.quarter.unwrap_or(1) as f32,
                r.weekno.unwrap_or(0) as f32,
                // [9-17] Percentage-based OHLC
                r.open_pct.unwrap_or(0.0) as f32,
                r.high_pct.unwrap_or(0.0) as f32,
                r.low_pct.unwrap_or(0.0) as f32,
                r.close_pct.unwrap_or(0.0) as f32,
                r.high_from_open_pct.unwrap_or(0.0) as f32,
                r.low_from_open_pct.unwrap_or(0.0) as f32,
                r.close_from_open_pct.unwrap_or(0.0) as f32,
                r.intraday_range_pct.unwrap_or(0.0) as f32,
                r.close_position_in_range.unwrap_or(0.5) as f32,
                // [18-25] Moving averages (percentage-based)
                r.ema_5_pct.unwrap_or(0.0) as f32,
                r.ema_10_pct.unwrap_or(0.0) as f32,
                r.ema_20_pct.unwrap_or(0.0) as f32,
                r.ema_30_pct.unwrap_or(0.0) as f32,
                r.ema_60_pct.unwrap_or(0.0) as f32,
                r.sma_5.unwrap_or(0.0) as f32,
                r.sma_10.unwrap_or(0.0) as f32,
                r.sma_20.unwrap_or(0.0) as f32,
                // [26-32] MACD
                r.macd_line.unwrap_or(0.0) as f32,
                r.macd_signal.unwrap_or(0.0) as f32,
                r.macd_histogram.unwrap_or(0.0) as f32,
                r.macd_weekly_line.unwrap_or(0.0) as f32,
                r.macd_weekly_signal.unwrap_or(0.0) as f32,
                r.macd_monthly_line.unwrap_or(0.0) as f32,
                r.macd_monthly_signal.unwrap_or(0.0) as f32,
                // [33-36] Technical
                r.rsi_14.unwrap_or(50.0) as f32,
                r.kdj_k.unwrap_or(50.0) as f32,
                r.kdj_d.unwrap_or(50.0) as f32,
                r.kdj_j.unwrap_or(50.0) as f32,
                // [37-41] Bollinger Bands (percentage-based)
                r.bb_upper_pct.unwrap_or(0.0) as f32,
                r.bb_middle_pct.unwrap_or(0.0) as f32,
                r.bb_lower_pct.unwrap_or(0.0) as f32,
                r.bb_bandwidth.unwrap_or(0.0) as f32,
                r.bb_percent_b.unwrap_or(0.5) as f32,
                // [42-47] Volatility
                r.atr.unwrap_or(0.0) as f32,
                r.volatility_5.unwrap_or(0.0) as f32,
                r.volatility_20.unwrap_or(0.0) as f32,
                r.asi.unwrap_or(0.0) as f32,
                r.obv.unwrap_or(0.0) as f32,
                r.volume_ratio.unwrap_or(1.0) as f32,
                // [48-51] Momentum
                r.price_momentum_5.unwrap_or(0.0) as f32,
                r.price_momentum_10.unwrap_or(0.0) as f32,
                r.price_momentum_20.unwrap_or(0.0) as f32,
                r.price_position_52w.unwrap_or(0.5) as f32,
                // [52-54] Candlestick
                r.body_size.unwrap_or(0.0) as f32,
                r.upper_shadow.unwrap_or(0.0) as f32,
                r.lower_shadow.unwrap_or(0.0) as f32,
                // [55-62] Trend & strength
                r.trend_strength.unwrap_or(0.0) as f32,
                r.adx_14.unwrap_or(25.0) as f32,
                r.vwap_distance_pct.unwrap_or(0.0) as f32,
                r.cmf_20.unwrap_or(0.0) as f32,
                50.0 as f32,
                r.williams_r_14.unwrap_or(-50.0) as f32,
                r.aroon_up_25.unwrap_or(50.0) as f32,
                r.aroon_down_25.unwrap_or(50.0) as f32,
                // [63-65] Lagged returns
                r.return_lag_1.unwrap_or(0.0) as f32,
                r.return_lag_2.unwrap_or(0.0) as f32,
                r.return_lag_3.unwrap_or(0.0) as f32,
                // [66-67] Gap features
                r.overnight_gap.unwrap_or(0.0) as f32,
                r.gap_pct.unwrap_or(0.0) as f32,
                // [68-69] Volume features
                r.volume_roc_5.unwrap_or(0.0) as f32,
                if r.volume_spike.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                // [70-73] Price ROC
                r.price_roc_5.unwrap_or(0.0) as f32,
                r.price_roc_10.unwrap_or(0.0) as f32,
                r.price_roc_20.unwrap_or(0.0) as f32,
                r.hist_volatility_20.unwrap_or(0.0) as f32,
                // [74-77] Candlestick patterns
                if r.is_doji.unwrap_or(false) { 1.0 } else { 0.0 },
                if r.is_hammer.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                if r.is_shooting_star.unwrap_or(false) {
                    1.0
                } else {
                    0.0
                },
                r.consecutive_days.unwrap_or(0) as f32,
                // [78-80] Index CSI300 (percentage-based)
                r.index_csi300_pct_chg.unwrap_or(0.0) as f32,
                r.index_csi300_vs_ma5_pct.unwrap_or(0.0) as f32,
                r.index_csi300_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [81-83] Index Star50 (percentage-based)
                r.index_star50_pct_chg.unwrap_or(0.0) as f32,
                r.index_star50_vs_ma5_pct.unwrap_or(0.0) as f32,
                r.index_star50_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [84-86] Index ChiNext (percentage-based)
                r.index_chinext_pct_chg.unwrap_or(0.0) as f32,
                r.index_chinext_vs_ma5_pct.unwrap_or(0.0) as f32,
                r.index_chinext_vs_ma20_pct.unwrap_or(0.0) as f32,
                // [87-90] Money flow
                r.net_mf_vol.unwrap_or(0.0) as f32,
                r.net_mf_amount.unwrap_or(0.0) as f32,
                r.smart_money_ratio.unwrap_or(0.0) as f32,
                r.large_order_flow.unwrap_or(0.0) as f32,
                // [91-93] Industry
                r.industry_avg_return.unwrap_or(0.0) as f32,
                r.stock_vs_industry.unwrap_or(0.0) as f32,
                r.industry_momentum_5d.unwrap_or(0.0) as f32,
                // [94-95] Volatility regime
                r.vol_percentile.unwrap_or(0.5) as f32,
                if r.high_vol_regime.unwrap_or(0) != 0 {
                    1.0
                } else {
                    0.0
                },
                // [96-104] Reserved
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect();

    let last_date = records.last().map(|r| r.trade_date.clone());
    // build next_day_returns aligned with values
    let nd_returns: Vec<Option<f32>> = records.iter().map(|r| r.next_day_return.map(|v| v as f32)).collect();
    StockItem { values, last_trade_date: last_date.clone(), dataset_last_date: last_date, next_day_returns: nd_returns }
}
