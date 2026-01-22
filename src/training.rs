use burn::train::{LearnerBuilder, LearningStrategy};
use burn::train::metric::LossMetric;
use burn::tensor::backend::AutodiffBackend;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, Recorder};
use burn::module::Module;
use crate::model::ModelConfig;
use crate::dataset::DbStockDataset;
use crate::batcher::StockBatcher;
use crate::db::{DbClient, MlTrainingRecord};
// use yahoo_finance_api::Quote; // Disabled - not needed
use anyhow::Result;
use std::fs;

/*
// Yahoo Finance training disabled - use run_from_db instead
pub fn run<B: AutodiffBackend>(device: B::Device, ticker: &str, quotes: Vec<Quote>, ashr_quotes: Vec<Quote>) {
    // Config
    let seq_len = 32;
    let batch_size = 32;
    let num_epochs = 20; // Increased epochs
    let learning_rate = 5e-4;

    // Split data into train and validation (80/20)
    let split_idx = (quotes.len() as f32 * 0.8) as usize;
    let (train_data, valid_data) = quotes.split_at(split_idx);

    // Dataset with market data (China)
    // We request seq_len + 1 so we can split into input and target
    let train_dataset = RealStockDataset::with_market_data(ticker, train_data.to_vec(), ashr_quotes.clone(), seq_len + 1);
    let valid_dataset = RealStockDataset::with_market_data(ticker, valid_data.to_vec(), ashr_quotes, seq_len + 1);

    // Batcher
    let batcher_train = StockBatcher::<B>::new(device.clone());
    let batcher_valid = StockBatcher::<B::InnerBackend>::new(device.clone());

    // DataLoader - NO SHUFFLE for time series data!
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(4)
        .build(valid_dataset);

    // Model
    let model_config = ModelConfig::new();
    let model = model_config.init::<B>(&device);

    // Learner
    let artifact_dir = format!("{}/artifacts", std::env::current_dir().unwrap().display());
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .build(
            model,
            AdamConfig::new().init(),
            learning_rate,
            LearningStrategy::default(),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained.model
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved successfully");
}
*/

/// Train model using database records from ml_training_dataset table
pub async fn run_from_db<B: AutodiffBackend>(device: B::Device, db_client: &DbClient) -> Result<()> {
    println!("Fetching training data from database...");
    let train_records = db_client.fetch_training_data().await?;
    println!("Loaded {} training records", train_records.len());
    
    println!("Fetching validation data from database...");
    let valid_records = db_client.fetch_validation_data().await?;
    println!("Loaded {} validation records", valid_records.len());
    
    if train_records.is_empty() || valid_records.is_empty() {
        anyhow::bail!("No data found in ml_training_dataset table. Please run dataset_creator first.");
    }
    
    run_with_data::<B>(device, train_records, valid_records)
}

/// Train model with pre-fetched data (synchronous, works with GPU backends that aren't Send)
pub fn run_with_data<B: AutodiffBackend>(
    device: B::Device,
    train_records: Vec<MlTrainingRecord>,
    valid_records: Vec<MlTrainingRecord>,
) -> Result<()> {
    
    // Config - Optimized for CPU training
    let seq_len = 70; // Increased to 70 to capture longer patterns (~3.5 months)
    let batch_size = 32; // Reduced from 64 for better generalization
    let num_epochs = 30;
    let learning_rate = 3e-4; // Reduced from 5e-4 for more stable convergence

    // CRITICAL: Group records by ts_code and sort by trade_date to prevent sequences from crossing stock boundaries
    // The database returns data ordered by ts_code, trade_date, but we need to create separate datasets per stock
    println!("Grouping data by stock symbols...");
    let train_datasets = DbStockDataset::from_records_grouped(train_records, seq_len + 1);
    let valid_datasets = DbStockDataset::from_records_grouped(valid_records, seq_len + 1);
    
    println!("\\n=== Creating Combined Dataset ===");
    println!("Extracting sequences from {} training datasets...", train_datasets.len());
    
    // Create a combined dataset by concatenating all per-stock datasets
    // This ensures sequences never cross stock boundaries
    use burn::data::dataset::InMemDataset;
    
    let mut all_train_items = Vec::new();
    let mut seqs_per_stock = Vec::new();
    for dataset in &train_datasets {
        let dataset_len = dataset.len();
        seqs_per_stock.push(dataset_len);
        for i in 0..dataset_len {
            if let Some(item) = dataset.get(i) {
                all_train_items.push(item);
            }
        }
    }
    
    // Show distribution
    println!("Sequences per stock: min={}, max={}, avg={:.1}",
        seqs_per_stock.iter().min().unwrap_or(&0),
        seqs_per_stock.iter().max().unwrap_or(&0),
        if seqs_per_stock.is_empty() { 0.0 } else { seqs_per_stock.iter().sum::<usize>() as f64 / seqs_per_stock.len() as f64 }
    );
    
    let mut all_valid_items = Vec::new();
    for dataset in &valid_datasets {
        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                all_valid_items.push(item);
            }
        }
    }
    
    let train_dataset = InMemDataset::new(all_train_items);
    let valid_dataset = InMemDataset::new(all_valid_items);
    
    let train_len = train_dataset.len();
    let valid_len = valid_dataset.len();
    
    println!("Training dataset size: {} sequences", train_len);
    println!("Validation dataset size: {} sequences", valid_len);
    
    if train_len == 0 || valid_len == 0 {
        anyhow::bail!("No valid sequences found. Check if data has enough history.");
    }

    // Batcher
    let batcher_train = StockBatcher::<B>::new(device.clone());
    let batcher_valid = StockBatcher::<B::InnerBackend>::new(device.clone());

    // DataLoader - NO SHUFFLE for time series data!
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(4)
        .build(valid_dataset);

    // Model
    let model_config = ModelConfig::new();
    let model = model_config.init::<B>(&device);

    // Create checkpoint directory
    let artifact_dir = format!("{}/artifacts", std::env::current_dir().unwrap().display());
    let checkpoint_dir = format!("{}/checkpoints", artifact_dir);
    fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
    
    println!("\n=== Training Configuration ===");
    println!("Checkpoint directory: {}", checkpoint_dir);
    println!("Best model will be saved to: {}/best_model", checkpoint_dir);

    // Learner with checkpoint callback
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        // Add checkpoint callback to save best model
        .with_file_checkpointer(CompactRecorder::new())
        .build(
            model,
            AdamConfig::new().init(),
            learning_rate,
            LearningStrategy::default(),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save final model
    model_trained.model
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved successfully");
    
    // Find and copy best model based on validation loss
    println!("\n=== Finding Best Model ===");
    if let Ok(best_epoch) = find_best_epoch(&artifact_dir) {
        println!("Best validation loss found at epoch {}", best_epoch);
        
        // Copy best model checkpoint
        let best_model_src = format!("{}/valid/epoch-{}/model.mpk", artifact_dir, best_epoch);
        let best_model_dst = format!("{}/checkpoints/best_model.mpk", artifact_dir);
        
        if let Ok(_) = fs::copy(&best_model_src, &best_model_dst) {
            println!("✓ Best model saved to: {}", best_model_dst);
            
            // Also save metadata
            let metadata = format!("Best epoch: {}\nSaved from: {}\n", best_epoch, best_model_src);
            fs::write(format!("{}/checkpoints/best_model_info.txt", artifact_dir), metadata)
                .expect("Failed to write metadata");
        } else {
            println!("⚠ Warning: Could not copy best model from epoch {}", best_epoch);
        }
    } else {
        println!("⚠ Warning: Could not determine best epoch, using final model");
    }
    
    println!("Training completed. Final model saved to {}/model", artifact_dir);
    Ok(())
}

/// Find the epoch with the best (lowest) validation loss
fn find_best_epoch(artifact_dir: &str) -> Result<usize> {
    let valid_dir = format!("{}/valid", artifact_dir);
    let mut best_epoch = 1;
    let mut best_loss = f32::MAX;
    
    // Scan all epoch directories
    for epoch in 1..=100 {  // Check up to 100 epochs
        let loss_file = format!("{}/epoch-{}/Loss.log", valid_dir, epoch);
        if let Ok(content) = fs::read_to_string(&loss_file) {
            // Get the last line which contains the final validation loss for that epoch
            if let Some(last_line) = content.lines().last() {
                // Parse the loss value (format: "loss_value,timestamp")
                if let Some(loss_str) = last_line.split(',').next() {
                    if let Ok(loss) = loss_str.parse::<f32>() {
                        if loss < best_loss && loss > 0.0 && loss < 1000.0 {  // Ignore extreme values
                            best_loss = loss;
                            best_epoch = epoch;
                        }
                    }
                }
            }
        } else {
            // No more epochs found
            break;
        }
    }
    
    if best_loss < f32::MAX {
        println!("Best validation loss: {:.6} at epoch {}", best_loss, best_epoch);
        Ok(best_epoch)
    } else {
        anyhow::bail!("No valid epochs found")
    }
}
