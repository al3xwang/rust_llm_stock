use anyhow::Result;
use clap::Parser;
use tch::{nn, Device, Tensor};
use rust_llm_stock::{
    db,
    model_torch::{TorchStockModel, ModelConfig},
    dataset::DbStockDataset,
    feature_normalization::{normalize_features, denormalize_pct_feature, FEATURE_SIZE},
};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "test-pytorch-model")]
#[command(about = "Test PyTorch model accuracy on validation/test data")]
struct Cli {
    /// Model path (e.g., artifacts/best_model.safetensors)
    #[arg(short, long, default_value = "artifacts/best_model.safetensors")]
    model_path: String,
    
    /// Specific stock to test (optional, tests all if not specified)
    #[arg(short, long)]
    stock: Option<String>,
    
    /// Start date for test period (YYYYMMDD format)
    #[arg(long, default_value = "20240401")]
    start_date: String,
    
    /// End date for test period (YYYYMMDD format, default: today)
    #[arg(long)]
    end_date: Option<String>,
    
    /// Use GPU if available
    #[arg(long)]
    use_gpu: bool,
    
    /// Show individual predictions (for tomorrow's forecast)
    #[arg(long)]
    show_predictions: bool,
    
    /// Predict next day (forward prediction without validation)
    #[arg(long)]
    predict_next: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Database connection
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| {
            "host=localhost port=5432 user=postgres password=12341234 dbname=research".to_string()
        });
    
    println!("Connecting to database...");
    let db_client = db::DbClient::new(&db_url).await?;
    println!("✓ Database connected!");
    println!();
    
    // Device selection
    let device = if cli.use_gpu && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);
    
    // Load model
    println!("Loading model from: {}", cli.model_path);
    let mut vs = nn::VarStore::new(device);
    let config = ModelConfig::default();
    let model = TorchStockModel::new(&vs.root(), &config);
    vs.load(&cli.model_path)?;
    vs.freeze(); // Set to evaluation mode
    println!("✓ Model loaded successfully!");
    println!();
    
    // Fetch test data
    println!("=== Test Configuration ===");
    if let Some(ref stock) = cli.stock {
        println!("Stock: {}", stock);
    } else {
        println!("Testing: Validation dataset");
    }
    println!();
    
    let records = if let Some(ref stock) = cli.stock {
        let end_date = cli.end_date.unwrap_or_else(|| {
            chrono::Local::now().format("%Y%m%d").to_string()
        });
        println!("Period: {} to {}", cli.start_date, end_date);
        
        // Try fetching from ml_training_dataset first
        let records = db_client.fetch_stock_data(stock, &cli.start_date, &end_date).await?;
        
        // If no data found, calculate features on-the-fly from stock_daily
        if records.is_empty() {
            println!("⚠ Stock not found in ml_training_dataset");
            println!("📊 Calculating features on-the-fly from stock_daily...");
            db_client.fetch_and_calculate_features(stock, &cli.start_date, &end_date).await?
        } else {
            records
        }
    } else {
        println!("Loading validation dataset from database...");
        db_client.fetch_validation_data().await?
    };
    
    println!("Loaded {} records", records.len());
    println!();
    
    // Run evaluation
    let seq_len = 30;  // Must match training sequence length
    let metrics = evaluate_model(&model, records.clone(), device, seq_len, cli.show_predictions)?;
    
    // Print results
    println!("\n╔════════════════════════════════════════════════╗");
    println!("║              Test Results                      ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("Total Sequences Evaluated: {}", metrics.total_predictions);
    println!();
    println!("Overall Metrics (Normalized Features):");
    println!("  MSE:  {:.6}", metrics.mse);
    println!("  RMSE: {:.6}", metrics.rmse);
    println!("  MAE:  {:.6}", metrics.mae);
    println!();
    println!("═══════════════════════════════════════════════");
    println!("  🎯 DIRECTION ACCURACY: {:.2}%", metrics.direction_accuracy * 100.0);
    println!("═══════════════════════════════════════════════");
    println!();
    println!("Confidence-Based Filtering (threshold ≥35%):");
    println!("  Coverage: {:.1}%", metrics.high_conf_percentage * 100.0);
    println!("  Accuracy: {:.2}%", metrics.high_conf_accuracy * 100.0);
    println!();
    
    // Make forward prediction if requested
    if cli.predict_next && cli.stock.is_some() {
        println!("\n╔════════════════════════════════════════════════╗");
        println!("║          Forward Prediction (Next Day)        ║");
        println!("╚════════════════════════════════════════════════╝");
        println!();
        
        if let Some(prediction) = predict_next_day(&model, records, device, seq_len)? {
            let direction_str = if prediction.predicted_direction { "↑ UP" } else { "↓ DOWN" };
            let direction_emoji = if prediction.predicted_direction { "📈" } else { "📉" };
            
            println!("Stock: {}", prediction.stock_code);
            println!("Last Trading Day: {}", prediction.last_date);
            println!();
            println!("{} Predicted Direction: {} {}", direction_emoji, direction_str, direction_emoji);
            println!("   Expected Change: {:+.2}%", prediction.predicted_pct_change);
            println!("   Confidence: {:.1}%", prediction.confidence * 100.0);
            println!();
            
            if prediction.confidence >= 0.35 {
                println!("✓ High confidence prediction (≥35%)");
            } else {
                println!("⚠ Low confidence - use with caution");
            }
            println!();
        } else {
            println!("⚠ Not enough data to make a forward prediction");
            println!("  (Need at least {} days of historical data)", seq_len);
            println!();
        }
    }
    
    Ok(())
}

struct NextDayPrediction {
    stock_code: String,
    last_date: String,
    predicted_direction: bool,
    predicted_pct_change: f32,
    confidence: f64,
}

fn predict_next_day(
    model: &TorchStockModel,
    records: Vec<db::MlTrainingRecord>,
    device: Device,
    seq_len: usize,
) -> Result<Option<NextDayPrediction>> {
    if records.len() < seq_len {
        return Ok(None);
    }
    
    // Sort by date
    let mut sorted_records = records.clone();
    sorted_records.sort_by_key(|r| r.trade_date.clone());
    
    let stock_code = sorted_records[0].ts_code.clone();
    let last_date = sorted_records.last().unwrap().trade_date.clone();
    
    // Create dataset with seq_len + 1 to get one sequence
    let datasets = DbStockDataset::from_records_grouped(sorted_records, seq_len + 1);
    
    if datasets.is_empty() {
        return Ok(None);
    }
    
    // Get the last sequence (most recent data)
    let dataset = &datasets[0];
    let last_idx = dataset.len().saturating_sub(1);
    
    if let Some(item) = dataset.get(last_idx) {
        // Prepare single-item batch
        if let Ok((inputs, _)) = prepare_batch(&[item], device, seq_len) {
            // Make prediction
            let (outputs, confidence) = tch::no_grad(|| {
                model.forward_with_confidence(&inputs, false)
            });
            
            // Extract the prediction for the next day (last position in sequence)
            let normalized_pred = outputs.get(0).get((seq_len - 1) as i64).get(15).double_value(&[]) as f32;
            let pred_pct_change = denormalize_pct_feature(normalized_pred);
            let conf_score = confidence.get(0).get((seq_len - 1) as i64).get(0).double_value(&[]).max(0.0).min(1.0);
            let predicted_direction = pred_pct_change > 0.0;
            
            return Ok(Some(NextDayPrediction {
                stock_code,
                last_date,
                predicted_direction,
                predicted_pct_change: pred_pct_change,
                confidence: conf_score,
            }));
        }
    }
    
    Ok(None)
}

struct TestMetrics {
    total_predictions: usize,
    mse: f64,
    rmse: f64,
    mae: f64,
    close_mae: f64,
    close_rmse: f64,
    direction_accuracy: f64,
    high_conf_accuracy: f64,
    high_conf_percentage: f64,
}

fn evaluate_model(
    model: &TorchStockModel,
    records: Vec<db::MlTrainingRecord>,
    device: Device,
    seq_len: usize,
    show_predictions: bool,
) -> Result<TestMetrics> {
    // Group by stock
    let mut grouped: HashMap<String, Vec<db::MlTrainingRecord>> = HashMap::new();
    for record in records.clone() {
        grouped.entry(record.ts_code.clone())
            .or_insert_with(Vec::new)
            .push(record);
    }
    
    // Sort each group by date
    for records in grouped.values_mut() {
        records.sort_by_key(|r| r.trade_date.clone());
    }
    
    // Create datasets
    let datasets = DbStockDataset::from_records_grouped(
        grouped.into_values().flatten().collect(),
        seq_len + 1
    );
    
    // Collect all items and track metadata
    let mut items = Vec::new();
    let mut item_metadata: Vec<(String, String)> = Vec::new(); // (stock_code, date)
    
    for (stock_code, stock_records) in records.iter()
        .fold(HashMap::new(), |mut map: HashMap<String, Vec<&db::MlTrainingRecord>>, r| {
            map.entry(r.ts_code.clone()).or_insert_with(Vec::new).push(r);
            map
        })
        .into_iter()
    {
        let mut sorted_records: Vec<_> = stock_records.into_iter().collect();
        sorted_records.sort_by_key(|r| &r.trade_date);
        
        // Track which dates correspond to each sequence
        for i in 0..sorted_records.len().saturating_sub(seq_len) {
            if let Some(last_record) = sorted_records.get(i + seq_len) {
                item_metadata.push((stock_code.clone(), last_record.trade_date.clone()));
            }
        }
    }
    
    for dataset in &datasets {
        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                items.push(item);
            }
        }
    }
    
    println!("Evaluating {} sequences...", items.len());
    
    let mut total_mse = 0.0;
    let mut total_mae = 0.0;
    let mut total_close_abs_error = 0.0;
    let mut total_close_sq_error = 0.0;
    let mut correct_directions = 0;
    let mut total_directions = 0;
    let mut high_confidence_correct = 0;
    let mut high_confidence_total = 0;
    let mut actual_predictions = 0;
    let mut total_confidence_score = 0.0_f64;
    let mut min_confidence = 1.0_f64;
    let mut max_confidence = 0.0_f64;
    
    // Lower confidence threshold for better coverage
    // Most predictions should be above this if confidence head is well-calibrated
    let confidence_threshold = 0.35; // Lowered to 0.35 for broader capture
    
    // Evaluate in batches
    let batch_size = 32;
    tch::no_grad(|| {
        for batch_start in (0..items.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(items.len());
            let batch = &items[batch_start..batch_end];
            
            if let Ok((inputs, targets)) = prepare_batch(batch, device, seq_len) {
                let (outputs, confidence) = model.forward_with_confidence(&inputs, false); // false = evaluation mode
                
                // Calculate normalized losses
                let diff = &outputs - &targets;
                let mse = (&diff * &diff).mean(tch::Kind::Float).double_value(&[]);
                if !mse.is_nan() && mse.is_finite() {
                    total_mse += mse * batch.len() as f64;
                }
                
                let mae = diff.abs().mean(tch::Kind::Float).double_value(&[]);
                if !mae.is_nan() && mae.is_finite() {
                    total_mae += mae * batch.len() as f64;
                }
                
                // Direction accuracy using pct_change (feature 15)
                for i in 0..batch.len() {
                    if i >= batch[i].values.len() - 1 { continue; }
                    
                    // Get pct_change predictions and actuals at last timestep
                    let normalized_pred = outputs.get(i as i64).get((seq_len - 1) as i64).get(15).double_value(&[]) as f32;
                    let pred_pct_change = denormalize_pct_feature(normalized_pred);
                    let actual_pct_change = batch[i].values[seq_len][15]; // pct_change at next timestep
                    
                    // Track close price errors for reporting
                    // Use pre_close from last input timestep as the base price
                    let pre_close = batch[i].values[seq_len - 1][6]; // pre_close at last input timestep
                    let predicted_close = pre_close * (1.0 + pred_pct_change / 100.0); // Apply predicted pct change
                    let actual_close = pre_close * (1.0 + actual_pct_change / 100.0); // Apply actual pct change
                    
                    // Calculate close price errors
                    let close_error = (predicted_close - actual_close).abs();
                    if close_error.is_finite() && !close_error.is_nan() {
                        total_close_abs_error += close_error as f64;
                        total_close_sq_error += (close_error * close_error) as f64;
                        actual_predictions += 1;
                    }
                    
                    // Get confidence score (clamp to 0-1 range for safety)
                    let raw_conf = confidence.get(i as i64).get((seq_len - 1) as i64).get(0).double_value(&[]);
                    let conf_score = raw_conf.max(0.0).min(1.0); // Ensure 0-1 range
                    
                    // Track confidence statistics
                    total_confidence_score += conf_score;
                    min_confidence = min_confidence.min(conf_score);
                    max_confidence = max_confidence.max(conf_score);
                    
                    // Direction from pct_change: positive = up, negative = down
                    let predicted_direction = pred_pct_change > 0.0;
                    let actual_direction = actual_pct_change > 0.0;
                    let is_correct = predicted_direction == actual_direction;
                    
                    // Show individual predictions if requested (last 5 predictions = most recent)
                    if show_predictions && batch_start + i >= items.len().saturating_sub(5) {
                        let idx = batch_start + i;
                        let (stock_code, trade_date) = item_metadata.get(idx)
                            .map(|(s, d)| (s.as_str(), d.as_str()))
                            .unwrap_or(("", ""));
                        let direction_str = if predicted_direction { "↑ UP" } else { "↓ DOWN" };
                        let actual_str = if actual_direction { "↑" } else { "↓" };
                        let correct_mark = if is_correct { "✓" } else { "✗" };
                        println!("  {} {} | {} ({:+.2}%) | Actual: {} {} | Conf: {:.1}%",
                            trade_date, stock_code, direction_str, pred_pct_change, actual_str, correct_mark, conf_score * 100.0);
                    }
                    
                    if is_correct {
                        correct_directions += 1;
                    }
                    total_directions += 1;
                    
                    // Track high-confidence predictions
                    if conf_score >= confidence_threshold {
                        high_confidence_total += 1;
                        if is_correct {
                            high_confidence_correct += 1;
                        }
                    }
                }
            }
            
            // Progress
            if (batch_end % 1000) == 0 || batch_end == items.len() {
                print!("\rProgress: {}/{}", batch_end, items.len());
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
    });
    
    println!(); // New line after progress
    
    let total_count = actual_predictions.max(1) as f64;
    let mse = total_mse / items.len() as f64;
    let mae = total_mae / items.len() as f64;
    let close_mae = total_close_abs_error / total_count;
    let close_rmse = (total_close_sq_error / total_count).sqrt();
    let direction_accuracy = if total_directions > 0 {
        correct_directions as f64 / total_directions as f64
    } else {
        0.0
    };
    
    let high_conf_accuracy = if high_confidence_total > 0 {
        high_confidence_correct as f64 / high_confidence_total as f64
    } else {
        0.0
    };
    
    let high_conf_percentage = if total_directions > 0 {
        high_confidence_total as f64 / total_directions as f64
    } else {
        0.0
    };
    
    let avg_confidence = if total_directions > 0 {
        total_confidence_score / total_directions as f64
    } else {
        0.0
    };
    
    println!("\nConfidence Statistics:");
    println!("  Average: {:.4}", avg_confidence);
    println!("  Min: {:.4}", min_confidence);
    println!("  Max: {:.4}", max_confidence);
    println!("  Threshold: {:.2}", confidence_threshold);
    
    Ok(TestMetrics {
        total_predictions: actual_predictions,
        mse,
        rmse: mse.sqrt(),
        mae,
        close_mae,
        close_rmse,
        direction_accuracy,
        high_conf_accuracy,
        high_conf_percentage,
    })
}

fn prepare_batch(
    batch: &[rust_llm_stock::dataset::StockItem],
    device: Device,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let mut input_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    let mut target_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    
    for item in batch {
        if item.values.len() < seq_len + 1 {
            continue;
        }
        
        // Extract reference price from the last input timestep (Day 29 in a 30-day window)
        // This avoids issues with lifetime-adjusted prices from far history
        let reference_close = item.values[seq_len - 1][6].max(0.01); // pre_close at last input day
        
        // Input: first seq_len timesteps (normalized relative to last day)
        for t in 0..seq_len {
            let normalized = normalize_features(item.values[t], reference_close);
            input_data.extend_from_slice(&normalized);
        }
        
        // Target: last seq_len timesteps (shifted by 1, normalized relative to last input day)
        for t in 1..=seq_len {
            let normalized = normalize_features(item.values[t], reference_close);
            target_data.extend_from_slice(&normalized);
        }
    }
    
    let inputs = Tensor::from_slice(&input_data)
        .reshape(&[batch_size as i64, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);
    
    let targets = Tensor::from_slice(&target_data)
        .reshape(&[batch_size as i64, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);
    
    Ok((inputs, targets))
}

// normalize_features provided via rust_llm_stock::feature_normalization
