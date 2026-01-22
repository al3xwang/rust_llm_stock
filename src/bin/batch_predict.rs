use anyhow::Result;
use clap::Parser;
use futures::stream::{self, StreamExt};
use rust_llm_stock::{
    dataset::DbStockDataset,
    db,
    feature_normalization::{FEATURE_SIZE, normalize_features},
    model_torch::{ModelConfig, TorchStockModel},
};
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Tensor};
use tokio::sync::Semaphore;

#[derive(Parser)]
#[command(name = "batch-predict")]
#[command(about = "Batch predict all stocks in ml_training_dataset and store results")]
struct Cli {
    /// Model path (e.g., artifacts/best_model.safetensors)
    #[arg(short, long, default_value = "artifacts/best_model.safetensors")]
    model_path: String,

    /// Model version/name to store with predictions
    #[arg(long, default_value = "pytorch_v1.0")]
    model_version: String,

    /// Use GPU if available
    #[arg(long)]
    use_gpu: bool,

    /// Concurrency level for DB fetch/inference (default: min(24, num_cpus*2))
    #[arg(long)]
    concurrency: Option<usize>,

    /// Limit number of stocks to process (for testing)
    #[arg(long)]
    limit: Option<usize>,

    /// Skip stocks (offset for parallel processing)
    #[arg(long, default_value = "0")]
    offset: usize,

    /// Lookback window in days for fetching historical features (should be >= seq_len)
    #[arg(long, default_value_t = 400)]
    lookback_days: i64,

    /// Start date for predictions (YYYYMMDD) - if not set, uses earliest available
    #[arg(long)]
    start_date: Option<String>,

    /// End date for predictions (YYYYMMDD) - if not set, uses latest available
    #[arg(long)]
    end_date: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Batch Stock Prediction System                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Database connection
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
        "host=localhost port=5432 user=postgres password=12341234 dbname=research".to_string()
    });

    println!("Connecting to database...");
    let db_client = db::DbClient::new(&db_url).await?;
    println!("✓ Database connected!");
    println!();

    // Device selection
    let device = if cli.use_gpu && tch::Cuda::is_available() {
        println!("Using GPU (CUDA)");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    println!();

    // Load model
    println!("Loading model from {}...", cli.model_path);
    let model_config = ModelConfig::default();
    let mut vs = tch::nn::VarStore::new(device);
    let model = TorchStockModel::new(&vs.root(), &model_config);

    // Try to load model - handle both old and new architectures
    match vs.load(&cli.model_path) {
        Ok(_) => {
            println!("✓ Model loaded successfully!");
        }
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("output_proj_3day") || err_str.contains("confidence_head_3day") {
                // Old single-output model - reinitialize dual heads with small weights
                // This allows forward() to work with backward compatibility
                println!("⚠️  Loading old single-output model (dual heads not present)");
                println!("   Model will use 1-day predictions only");
                println!("   (Consider retraining with dual-output architecture)");
                vs.load(&cli.model_path)?;
            } else {
                return Err(anyhow::anyhow!("Failed to load model: {}", e));
            }
        }
    }
    let model = Arc::new(model); // Wrap in Arc for sharing across tasks
    println!();

    // Get all stocks
    println!("Fetching list of stocks from ml_training_dataset...");
    // Performance: fetch symbols from stock_basic (listed stocks) instead of DISTINCT over ml_training_dataset
    // This avoids scanning ~13M rows and speeds up startup significantly
    let all_stocks = db_client.get_active_stocks().await?;
    println!("✓ Found {} active listed stocks", all_stocks.len());
    println!();

    // Apply offset and limit
    let stocks_to_process: Vec<String> = all_stocks
        .into_iter()
        .skip(cli.offset)
        .take(cli.limit.unwrap_or(usize::MAX))
        .collect();

    println!(
        "Processing {} stocks (offset: {})",
        stocks_to_process.len(),
        cli.offset
    );
    if let Some(limit) = cli.limit {
        println!("  (limited to {} stocks for testing)", limit);
    }
    println!();

    // Process each stock with parallel batching
    let seq_len = 60; // Must match training/test sequence length
    let lookback_days = cli.lookback_days.max((seq_len as i64) + 1);
    let mut total_predictions = 0;
    let mut total_saved = 0;
    let mut total_inserted = 0;
    let mut total_updated = 0;
    let mut total_no_data = 0;
    let _total_low_confidence = 0; // Reserved for future confidence filtering
    let mut failed_stocks = Vec::new();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║              Starting Batch Prediction                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Optimize batch size based on device
    // Moderate size for faster database commits and early result visibility
    let batch_size = if cli.use_gpu && tch::Cuda::is_available() {
        512 // GPU: moderate batches for faster commits to database
    } else {
        256 // CPU: increased for better parallelism
    };

    let total_batches = (stocks_to_process.len() + batch_size - 1) / batch_size;

    // Concurrency: maximize parallel DB operations for speed
    // With 88 cores, use 2x CPU count for optimal DB prefetch overlap with GPU inference
    let default_concurrency = (num_cpus::get() * 2).min(176);
    let concurrency = cli.concurrency.unwrap_or(default_concurrency);
    let db_semaphore = Arc::new(Semaphore::new(concurrency));

    println!(
        "Processing in {} batches of {} stocks each (device: {:?})",
        total_batches, batch_size, device
    );
    println!("  (Max {} concurrent DB ops)", concurrency);
    println!();

    for batch_idx in 0..total_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = ((batch_idx + 1) * batch_size).min(stocks_to_process.len());
        let batch_stocks = &stocks_to_process[batch_start..batch_end];
        let stocks_remaining = stocks_to_process.len() - batch_end;

        println!();
        println!("╔══════════════════════════════════════════════════════════╗");
        println!(
            "║  Batch {}/{}: {} stocks | {} remaining            ",
            batch_idx + 1,
            total_batches,
            batch_stocks.len(),
            stocks_remaining
        );
        println!("╚══════════════════════════════════════════════════════════╝");

        // Process batch with shared DB connection and GPU batching
        let mut batch_failures: Vec<String> = Vec::new();
        let mut stock_data = Vec::new();
        let mut no_data_in_batch = 0usize;

        // Fetch data for all stocks in parallel but limit concurrent DB connections
        use std::sync::atomic::{AtomicUsize, Ordering};
        let processed_in_batch = Arc::new(AtomicUsize::new(0));
        let batch_size_total = batch_stocks.len();

        let fetch_results: Vec<(String, Result<Vec<(String, Vec<f32>)>>)> =
            stream::iter(batch_stocks.iter())
                .map(|stock| {
                    let stock = stock.clone();
                    let semaphore = Arc::clone(&db_semaphore);
                    let db = &db_client;
                    let lookback_days = lookback_days;
                    let start_date = cli.start_date.clone();
                    let end_date = cli.end_date.clone();
                    let processed = Arc::clone(&processed_in_batch);
                    let batch_total = batch_size_total;
                    async move {
                        // Acquire semaphore permit before database operation
                        let _permit = semaphore.acquire().await.unwrap();
                        let result = prepare_stock_data(
                            db,
                            &stock,
                            seq_len,
                            lookback_days,
                            start_date,
                            end_date,
                        )
                        .await;

                        // Progress tracking
                        let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
                        let remaining = batch_total - count;
                        print!(
                            "\r  📊 Processing: {} | {}/{} done | {} left     ",
                            stock, count, batch_total, remaining
                        );
                        std::io::Write::flush(&mut std::io::stdout()).ok();

                        (stock, result)
                    }
                })
                .buffer_unordered(concurrency) // Process up to N stocks concurrently
                .collect()
                .await;

        println!(); // New line after progress updates

        // Process fetch results - flatten Vec of predictions per stock
        let mut stocks_with_data = 0;
        for (stock, result) in fetch_results {
            match result {
                Ok(predictions) if !predictions.is_empty() => {
                    stocks_with_data += 1;
                    // Add all date-specific predictions for this stock
                    for pred in predictions {
                        stock_data.push((stock.clone(), pred));
                    }
                }
                Ok(_) => {
                    no_data_in_batch += 1;
                }
                Err(e) => {
                    eprintln!("\n✗ {}: {:?}", stock, e);
                    batch_failures.push(stock);
                }
            }
        }

        // Show data collection summary
        println!(
            "  📦 Data collection: {} stocks with data | {} empty | {} errors",
            stocks_with_data,
            no_data_in_batch,
            batch_failures.len()
        );

        // Memory usage estimate for this batch
        let batch_memory_mb =
            (stock_data.len() * seq_len * FEATURE_SIZE * 4) as f64 / (1024.0 * 1024.0);
        if batch_memory_mb > 100.0 {
            println!("  ⚠️  Batch memory usage: ~{:.1} MB", batch_memory_mb);
        }

        // Batch inference on GPU/CPU
        if !stock_data.is_empty() {
            println!(
                "  🔮 Running inference on {} stock-date pairs...",
                stock_data.len()
            );
            match batch_infer_stocks(&*model, &stock_data, device, seq_len).await {
                Ok(predictions) => {
                    println!(
                        "  ✓ Inference complete: {} predictions generated",
                        predictions.len()
                    );
                    let mut upserts: Vec<db::PredictionInsert> =
                        Vec::with_capacity(stock_data.len());

                    for (i, (stock, (trade_date, _))) in stock_data.iter().enumerate() {
                        if let Some(&(
                            pred_return_1day,
                            confidence_1day,
                            pred_return_3day,
                            _confidence_3day,
                        )) = predictions.get(i)
                        {
                            upserts.push(db::PredictionInsert {
                                ts_code: stock.clone(),
                                trade_date: trade_date.clone(),
                                predicted_direction: pred_return_1day > 0.0,
                                predicted_return: pred_return_1day,
                                confidence: confidence_1day,
                                model_version: cli.model_version.clone(),
                                predicted_3day_return: Some(pred_return_3day),
                                predicted_3day_direction: Some(pred_return_3day > 0.0),
                            });
                        }
                    }

                    if !upserts.is_empty() {
                        // De-duplicate within the batch to avoid ON CONFLICT hitting same key twice
                        let mut unique: HashMap<(String, String, String), db::PredictionInsert> =
                            HashMap::with_capacity(upserts.len());
                        for upsert in upserts.into_iter() {
                            let key = (
                                upsert.ts_code.clone(),
                                upsert.trade_date.clone(),
                                upsert.model_version.clone(),
                            );
                            unique.insert(key, upsert);
                        }
                        let deduped: Vec<db::PredictionInsert> = unique.into_values().collect();
                        if deduped.len() < stock_data.len() {
                            println!(
                                "  ⚠️  Deduped {} duplicate predictions in this batch",
                                stock_data.len().saturating_sub(deduped.len())
                            );
                        }

                        println!("  💾 Saving {} predictions to database...", deduped.len());
                        match db_client.save_predictions_batch(&deduped).await {
                            Ok((inserted, updated)) => {
                                total_predictions += deduped.len();
                                total_saved += deduped.len();
                                total_inserted += inserted;
                                total_updated += updated;
                                println!("  ✓ Saved: {} new, {} updated", inserted, updated);
                            }
                            Err(e) => {
                                eprintln!("  ✗ Batch save failed: {:?}", e);
                                for (stock, _) in &stock_data {
                                    batch_failures.push(stock.clone());
                                }
                            }
                        }
                    } else {
                        println!("  ⚠️  No predictions to save (upserts vector is empty)");
                    }
                }
                Err(e) => {
                    eprintln!("Batch inference failed: {:?}", e);
                    for (stock, _) in &stock_data {
                        batch_failures.push(stock.clone());
                    }
                }
            }
        } else {
            println!("  ⚠️  No stock data to process in this batch (stock_data is empty)");
        }

        total_no_data += no_data_in_batch;
        failed_stocks.extend(batch_failures);

        // Explicitly free memory before next batch
        stock_data.clear();
        stock_data.shrink_to_fit();

        println!();
        println!(
            "✅ Batch {}/{} complete: Processed {}/{} stocks | {} predictions saved",
            batch_idx + 1,
            total_batches,
            batch_end,
            stocks_to_process.len(),
            total_saved
        );
    }

    // Summary
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                  Prediction Summary                     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Total stocks processed: {}", stocks_to_process.len());
    println!("Total predictions made: {}", total_predictions);
    println!("Total predictions saved: {}", total_saved);
    println!("  - First prediction for stock/date: {}", total_inserted);
    println!(
        "  - Additional predictions (tracking changes): {}",
        total_updated
    );
    println!("\nSkipped stocks:");
    println!("  - No data available: {}", total_no_data);
    println!("  - Low confidence: 0 (confidence filter disabled)");
    println!("  - Failed with errors: {}", failed_stocks.len());

    if !failed_stocks.is_empty() {
        println!();
        println!("Failed stocks:");
        for stock in &failed_stocks {
            println!("  - {}", stock);
        }
    }

    println!();
    println!("✓ Batch prediction completed!");
    println!();
    println!(
        "Note: Predictions are upserted per stock/date/model_version; reruns overwrite the latest values while keeping actuals."
    );
    println!();
    println!("Query predictions:");
    println!(
        "  psql -d research -c \"SELECT ts_code, trade_date, TO_CHAR(prediction_date, 'HH24:MI:SS') as time, predicted_return, confidence FROM stock_predictions ORDER BY prediction_date DESC LIMIT 30\""
    );
    println!();

    Ok(())
}

/// Prepare stock data for batch inference (fetch and normalize)
/// Returns Vec of (trade_date, normalized_features) for all dates in range
async fn prepare_stock_data(
    db_client: &db::DbClient,
    ts_code: &str,
    seq_len: usize,
    lookback_days: i64,
    start_date_param: Option<String>,
    end_date_param: Option<String>,
) -> Result<Vec<(String, Vec<f32>)>> {
    // Get latest trade date for this stock
    let latest_date = match db_client.get_latest_trade_date(ts_code).await? {
        Some(date) => date,
        None => return Ok(Vec::new()),
    };

    // Use provided end_date or latest available
    let end_date = end_date_param.unwrap_or(latest_date);

    // If start_date_param is provided, use it; otherwise calculate from end_date
    let prediction_start_date = start_date_param.unwrap_or_else(|| {
        // Default: predict only the last sequence (backward compatible)
        end_date.clone()
    });

    // Calculate fetch start date: lookback from the EARLIEST of (prediction_start_date or end_date)
    // We need extra history to calculate features for prediction_start_date
    let fetch_start_parsed = chrono::NaiveDate::parse_from_str(&prediction_start_date, "%Y%m%d")?;
    let fetch_start_with_buffer = fetch_start_parsed - chrono::Duration::days(lookback_days);
    let start_date = fetch_start_with_buffer.format("%Y%m%d").to_string();

    // Fetch historical data
    let records = match db_client
        .fetch_stock_data_for_prediction(ts_code, &start_date, &end_date)
        .await
    {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            println!(
                "  ⚠️  {}: No records in date range {}-{}",
                ts_code, start_date, end_date
            );
            return Ok(Vec::new());
        }
        Err(e) => {
            println!("  ✗ {}: DB fetch error: {:?}", ts_code, e);
            return Err(e);
        }
    };

    // Sort records in place (no clone)
    let mut sorted_records = records;
    sorted_records.sort_by(|a, b| a.trade_date.cmp(&b.trade_date));

    // Create dataset for single stock (no need for grouping)
    let dataset = DbStockDataset::new(sorted_records.clone(), seq_len + 1);
    if dataset.len() == 0 {
        println!(
            "  ⚠️  {}: Dataset empty (need seq_len+1={} days, had {} records)",
            ts_code,
            seq_len + 1,
            sorted_records.len()
        );
        return Ok(Vec::new());
    }
    let dataset_len = dataset.len();

    // Find the index in sorted_records where predictions should start
    // This is the first record >= prediction_start_date
    let prediction_start_idx = sorted_records
        .iter()
        .position(|r| r.trade_date >= prediction_start_date)
        .unwrap_or(sorted_records.len().saturating_sub(1));

    // Generate predictions for all sequences from prediction_start_date onwards
    // Pre-allocate with estimated capacity to reduce reallocations
    let dataset_start_idx = prediction_start_idx.saturating_sub(seq_len);
    let estimated_predictions = dataset_len.saturating_sub(dataset_start_idx).min(252); // Cap at ~1 year
    let mut all_predictions = Vec::with_capacity(estimated_predictions);

    // Dataset indices: each sequence in dataset[i] predicts sorted_records[i + seq_len]
    // So to predict records starting at prediction_start_idx, we need dataset sequences starting at:

    for i in dataset_start_idx..dataset_len {
        if let Some(item) = dataset.get(i) {
            // The target date for this sequence is at sorted_records[i + seq_len]
            let target_record_idx = i + seq_len;

            // Get the trade_date for this prediction
            let prediction_date = sorted_records
                .get(target_record_idx)
                .map(|r| r.trade_date.clone())
                .unwrap_or_else(|| end_date.clone());

            // Only include predictions >= prediction_start_date
            if prediction_date < prediction_start_date {
                continue;
            }

            // Normalize features
            let reference_close_pct = item.values[seq_len - 1][12].abs().max(0.01);

            // Pre-allocate exact size to avoid reallocations
            let mut input_data = Vec::with_capacity(seq_len * FEATURE_SIZE);
            unsafe {
                input_data.set_len(seq_len * FEATURE_SIZE);
            }

            for t in 0..seq_len {
                let normalized = normalize_features(item.values[t], reference_close_pct);
                let offset = t * FEATURE_SIZE;
                input_data[offset..offset + FEATURE_SIZE].copy_from_slice(&normalized);
            }

            all_predictions.push((prediction_date, input_data));
        }
    }

    // Explicitly drop dataset to free memory before returning
    drop(dataset);

    Ok(all_predictions)
}

/// Batch inference: process multiple stocks in one GPU forward pass
/// Returns: Vec<(pred_1day_return, conf_1day, pred_3day_return, conf_3day)>
async fn batch_infer_stocks(
    model: &TorchStockModel,
    stock_data: &[(String, (String, Vec<f32>))],
    device: Device,
    seq_len: usize,
) -> Result<Vec<(f64, f64, f64, f64)>> {
    let total = stock_data.len();
    if total == 0 {
        return Ok(Vec::new());
    }

    // Use smaller inference minibatches to avoid CUDA OOM
    let infer_batch = if matches!(device, Device::Cuda(_)) {
        128
    } else {
        256
    };
    let mut predictions = Vec::with_capacity(total);

    for chunk_start in (0..total).step_by(infer_batch) {
        let chunk_end = (chunk_start + infer_batch).min(total);
        let chunk = &stock_data[chunk_start..chunk_end];
        let chunk_size = chunk.len();

        // Concatenate all stock inputs into single tensor [chunk_size, seq_len, 105]
        let mut batch_input = Vec::with_capacity(chunk_size * seq_len * FEATURE_SIZE);
        for (_, (_, input_data)) in chunk {
            batch_input.extend_from_slice(input_data);
        }

        let input_tensor = Tensor::from_slice(&batch_input)
            .reshape(&[chunk_size as i64, seq_len as i64, FEATURE_SIZE as i64])
            .to_device(device);

        // Forward pass for this chunk
        let (output_1day, output_3day, conf_1day, conf_3day) =
            tch::no_grad(|| model.forward_dual(&input_tensor, false));

        // Extract predictions and confidences for each stock (both 1-day and 3-day)
        let output_1day_flat: Vec<f32> = output_1day.view([-1]).try_into()?;
        let output_3day_flat: Vec<f32> = output_3day.view([-1]).try_into()?;
        // conf_* shape is [batch, seq_len, 1] - flatten accordingly
        let conf_1day_flat: Vec<f32> = conf_1day.view([-1]).try_into()?;
        let conf_3day_flat: Vec<f32> = conf_3day.view([-1]).try_into()?;

        for i in 0..chunk_size {
            let stock_offset = i * seq_len * FEATURE_SIZE;
            let last_timestep_start = stock_offset + (seq_len - 1) * FEATURE_SIZE;

            // 1-day prediction (feature index 12)
            let predicted_return_1day_raw = output_1day_flat[last_timestep_start + 12] as f64;
            let predicted_return_1day = predicted_return_1day_raw.clamp(-15.0, 15.0);

            // 3-day prediction (feature index 12)
            let predicted_return_3day_raw = output_3day_flat[last_timestep_start + 12] as f64;
            let predicted_return_3day = predicted_return_3day_raw.clamp(-15.0, 15.0);

            // Confidence indices: each conf vector has length batch * seq_len * 1
            let conf_offset = i * seq_len + (seq_len - 1); // position of last timestep's confidence
            let confidence_1day = conf_1day_flat.get(conf_offset).copied().unwrap_or(0.0) as f64;
            let confidence_3day = conf_3day_flat.get(conf_offset).copied().unwrap_or(0.0) as f64;

            if predicted_return_1day.is_finite() && predicted_return_3day.is_finite() {
                // Log sample predictions for first few items in the first chunk
                if chunk_start == 0 && i < 3 {
                    println!(
                        "  Sample prediction: 1day={:.4}% (conf={:.4}), 3day={:.4}% (conf={:.4})",
                        predicted_return_1day,
                        confidence_1day,
                        predicted_return_3day,
                        confidence_3day
                    );
                }

                predictions.push((
                    predicted_return_1day,
                    confidence_1day,
                    predicted_return_3day,
                    confidence_3day,
                ));
            } else {
                predictions.push((0.0, 0.0, 0.0, 0.0));
            }
        }
    }

    Ok(predictions)
}
