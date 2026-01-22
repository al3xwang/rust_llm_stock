use crate::dataset::DbStockDataset;
use crate::db::MlTrainingRecord;
use crate::feature_normalization::{FEATURE_SIZE, normalize_features};
use crate::model_torch::{ModelConfig, TorchStockModel};
use anyhow::Result;
use rayon::prelude::*;
use std::fs;
use std::sync::mpsc;
use std::thread;
use tch::{Device, IndexOp, Tensor, nn, nn::OptimizerConfig};

pub fn train_with_torch(
    train_records: Vec<MlTrainingRecord>,
    valid_records: Vec<MlTrainingRecord>,
    device: Device,
    override_lr: Option<f64>,
    artifact_dir_override: Option<String>,
    batch_override: Option<usize>,
    weight_decay_override: Option<f64>,
    huber_delta: Option<f64>,
    dropout_override: Option<f32>,
) -> Result<()> {
    println!("Initializing PyTorch model on {:?}...", device);

    // Config - Optimized for 80%+ GPU utilization
    let seq_len = 60; // ~3 months of context, aligns with longer-horizon features
    // Defaults (can be overridden via CLI)
    let default_batch_size = 256; // Large batches to maximize GPU utilization
    let batch_size = batch_override.unwrap_or(default_batch_size);
    let max_epochs = 1000; // Maximum epochs (early stopping controls actual training)

    // Scale learning rate with batch size: LR_new = LR_old * sqrt(batch_size_new / batch_size_old)
    let default_learning_rate = 1e-4 * (batch_size as f64 / 48.0_f64).sqrt();
    let learning_rate = override_lr.unwrap_or(default_learning_rate);

    let early_stop_patience = 20; // More patience for finding minima
    let lr_decay_factor = 0.8; // Slightly gentler LR reduction for larger batch
    let lr_patience = 8; // Reduce LR earlier to help escape plateaus
    let weight_decay = weight_decay_override.unwrap_or(0.0); // L2 coefficient (added to loss)

    // Create datasets
    println!("Grouping data by stock symbols...");
    let train_datasets = DbStockDataset::from_records_grouped(train_records, seq_len + 1);
    let valid_datasets = DbStockDataset::from_records_grouped(valid_records, seq_len + 1);
    let total_train_sequences: usize = train_datasets.iter().map(|d| d.len()).sum();
    let total_valid_sequences: usize = valid_datasets.iter().map(|d| d.len()).sum();

    println!("Training sequences (streamed): {}", total_train_sequences);
    println!("Validation sequences (streamed): {}", total_valid_sequences);
    println!(
        "Estimated peak batch memory: ~{:.2} MB",
        (batch_size * seq_len * FEATURE_SIZE * 4) as f64 / (1024.0 * 1024.0)
    );

    // Create model
    let mut vs = nn::VarStore::new(device);
    let mut config = ModelConfig::default();
    if let Some(dr) = dropout_override {
        config.dropout = dr;
    }
    let model = TorchStockModel::new(&vs.root(), &config);

    // Optimizer
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // Create artifact directory (allow override via arg)
    let default_artifact_dir = format!("{}/artifacts", std::env::current_dir().unwrap().display());
    let artifact_dir = artifact_dir_override.unwrap_or(default_artifact_dir);
    fs::create_dir_all(&artifact_dir)?;

    println!("\n=== Training Configuration ===");
    println!("Device: {:?}", device);
    println!("Sequence length: {}", seq_len);
    println!("Batch size: {}", batch_size);
    println!(
        "Max epochs: {} (early stopping controls actual training)",
        max_epochs
    );
    println!("Learning rate: {}", learning_rate);
    println!(
        "Early stopping patience: {} epochs without improvement",
        early_stop_patience
    );
    println!("");

    let mut best_valid_loss = f64::MAX;
    let mut epochs_without_improvement = 0;
    let mut epochs_since_lr_decay = 0;
    let mut current_lr = learning_rate;

    // Training loop - controlled by patience-based early stopping
    let warmup_epochs = 5;

    for epoch in 1..=max_epochs {
        println!(
            "\nEpoch {} (early stopping patience: {}/{})",
            epoch, epochs_without_improvement, early_stop_patience
        );

        // Learning rate warmup
        if epoch <= warmup_epochs {
            let warmup_lr = learning_rate * (epoch as f64 / warmup_epochs as f64);
            opt.set_lr(warmup_lr);
            if epoch == warmup_epochs {
                println!("  🔥 Warmup complete, using full learning rate");
            }
        }

        // Training
        let (train_loss, train_mse, train_dir_loss) = train_epoch_stream(
            &model,
            &train_datasets,
            batch_size,
            &mut opt,
            &mut vs,
            device,
            seq_len,
            weight_decay,
        )?;
        println!(
            "  Train Loss: {:.6} (MSE: {:.6}, Dir: {:.6})",
            train_loss, train_mse, train_dir_loss
        );

        // Validation
        let valid_loss =
            validate_epoch_stream(&model, &valid_datasets, batch_size, device, seq_len)?; // validation excludes weight decay term
        println!("  Valid Loss: {:.6}", valid_loss);

        // Save best model and track improvement
        if valid_loss < best_valid_loss {
            best_valid_loss = valid_loss;
            epochs_without_improvement = 0;
            epochs_since_lr_decay = 0;
            let model_path = format!("{}/best_model.safetensors", artifact_dir);
            vs.save(&model_path)?;
            println!("  ✓ Saved best model (loss: {:.6})", valid_loss);
        } else {
            epochs_without_improvement += 1;
            epochs_since_lr_decay += 1;

            // Learning rate decay if plateauing
            if epochs_since_lr_decay >= lr_patience {
                current_lr *= lr_decay_factor;
                opt.set_lr(current_lr);
                epochs_since_lr_decay = 0;
                println!("  📉 Reducing learning rate to {:.6}", current_lr);
            }

            // Early stopping check
            if epochs_without_improvement >= early_stop_patience {
                println!(
                    "\n⏹  Early stopping triggered after {} epochs without improvement",
                    early_stop_patience
                );
                println!(
                    "  Best validation loss: {:.6} at epoch {}",
                    best_valid_loss,
                    epoch - early_stop_patience
                );
                break;
            }

            println!(
                "  ⚠  No improvement for {} epoch(s)",
                epochs_without_improvement
            );
        }

        // Save checkpoint
        if epoch % 10 == 0 {
            let checkpoint_path =
                format!("{}/checkpoint_epoch_{}.safetensors", artifact_dir, epoch);
            vs.save(&checkpoint_path)?;
        }
    }

    println!("\n✅ Training complete!");
    println!("Best validation loss: {:.6}", best_valid_loss);
    println!("Model saved to: {}/best_model.safetensors", artifact_dir);

    Ok(())
}

fn train_epoch_stream(
    model: &TorchStockModel,
    datasets: &[DbStockDataset],
    batch_size: usize,
    opt: &mut nn::Optimizer,
    vs: &mut nn::VarStore,
    device: Device,
    seq_len: usize,
    weight_decay: f64,
) -> Result<(f64, f64, f64)> {
    let mut total_loss = 0.0;
    let mut total_mse = 0.0;
    let mut total_dir_loss = 0.0;
    let mut num_batches = 0;

    // Dual-task learning weights
    let weight_1day_mse = 0.60; // 1-day MSE: 60%
    let weight_3day_mse = 0.25; // 3-day MSE: 25%
    let weight_direction = 0.15; // Direction loss: 15% (combined for both horizons)

    // Preallocate reusable device buffers to reduce allocation churn
    let mut dev_inputs_buf: Option<Tensor> = None;
    let mut dev_targets_buf: Option<Tensor> = None;

    for batch in StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len) {
        if let Ok((inputs_cpu, targets_cpu)) = batch {
            // Allocate or reuse device buffers with exact shapes
            let shape_inputs = inputs_cpu.size();
            let shape_targets = targets_cpu.size();

            // Guard against invalid feature dimension before indexing later
            if shape_inputs.len() != 3 || shape_targets.len() != 3 {
                eprintln!(
                    "⚠️  Skipping batch: unexpected tensor rank (inputs {:?}, targets {:?})",
                    shape_inputs, shape_targets
                );
                continue;
            }
            if shape_inputs[2] < FEATURE_SIZE as i64 || shape_targets[2] < FEATURE_SIZE as i64 {
                eprintln!(
                    "⚠️  Skipping batch: feature dim too small (got {}, expect >= {})",
                    shape_inputs[2], FEATURE_SIZE
                );
                continue;
            }

            // Guard against invalid feature dimension before indexing later
            if shape_inputs.len() != 3 || shape_targets.len() != 3 {
                eprintln!(
                    "⚠️  Skipping batch: unexpected tensor rank (inputs {:?}, targets {:?})",
                    shape_inputs, shape_targets
                );
                continue;
            }
            if shape_inputs[2] < FEATURE_SIZE as i64 || shape_targets[2] < FEATURE_SIZE as i64 {
                eprintln!(
                    "⚠️  Skipping batch: feature dim too small (got {}, expect >= {})",
                    shape_inputs[2], FEATURE_SIZE
                );
                continue;
            }

            if dev_inputs_buf.as_ref().map(|t| t.size()) != Some(shape_inputs.clone()) {
                dev_inputs_buf = Some(Tensor::zeros(&shape_inputs, (tch::Kind::Float, device)));
            }
            if dev_targets_buf.as_ref().map(|t| t.size()) != Some(shape_targets.clone()) {
                dev_targets_buf = Some(Tensor::zeros(&shape_targets, (tch::Kind::Float, device)));
            }

            let inputs = dev_inputs_buf.as_mut().unwrap();
            let targets = dev_targets_buf.as_mut().unwrap();

            // Copy CPU → GPU without reallocating device tensors
            inputs.copy_(&inputs_cpu);
            targets.copy_(&targets_cpu);

            // Debug: Check for NaN in inputs
            let inputs_has_nan = inputs.isnan().any().int64_value(&[]);
            let targets_has_nan = targets.isnan().any().int64_value(&[]);
            if inputs_has_nan != 0 || targets_has_nan != 0 {
                eprintln!(
                    "⚠️  Warning: NaN detected in batch data! Inputs: {}, Targets: {}",
                    inputs_has_nan != 0,
                    targets_has_nan != 0
                );
                continue; // Skip this batch
            }

            // Dual predictions from model
            let (pred_1day, pred_3day, _conf_1day, _conf_3day) = model.forward_dual(&inputs, true);

            // Check for NaN in dual outputs
            let pred_1day_has_nan = pred_1day.isnan().any().int64_value(&[]);
            let pred_3day_has_nan = pred_3day.isnan().any().int64_value(&[]);
            if pred_1day_has_nan != 0 || pred_3day_has_nan != 0 {
                eprintln!("⚠️  Warning: Dual predictions contain NaN!");
                continue; // Skip backward pass
            }

            // Prepare targets for both horizons
            let targets_1day = targets.shallow_clone();
            let targets_3day = targets.shallow_clone();

            // Multi-task learning loss
            // 1-day MSE (60% weight)
            let mse_loss_1day = pred_1day.mse_loss(&targets_1day, tch::Reduction::Mean);

            // 3-day MSE (25% weight)
            let mse_loss_3day = pred_3day.mse_loss(&targets_3day, tch::Reduction::Mean);

            // Direction loss for both horizons (averaged, 15% weight)
            let feature_index = 12i64;
            if pred_1day.size()[2] <= feature_index
                || pred_3day.size()[2] <= feature_index
                || targets.size()[2] <= feature_index
            {
                eprintln!(
                    "⚠️  Skipping batch: feature index {} out of bounds (pred_1day {:?}, pred_3day {:?}, targets {:?})",
                    feature_index,
                    pred_1day.size(),
                    pred_3day.size(),
                    targets.size()
                );
                continue;
            }
            let pred_pct_1day = pred_1day.i((.., .., feature_index));
            let pred_pct_3day = pred_3day.i((.., .., feature_index));
            let target_pct = targets.i((.., .., feature_index));

            let pred_direction_1day = pred_pct_1day.ge(0.0);
            let pred_direction_3day = pred_pct_3day.ge(0.0);
            let target_direction = target_pct.ge(0.0);

            let direction_loss_1day = (pred_direction_1day.to_kind(tch::Kind::Float)
                - target_direction.to_kind(tch::Kind::Float))
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

            let direction_loss_3day = (pred_direction_3day.to_kind(tch::Kind::Float)
                - target_direction.to_kind(tch::Kind::Float))
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

            let direction_loss = (direction_loss_1day.shallow_clone() + direction_loss_3day) * 0.5;

            // Extract scalar values BEFORE using in loss calculation (avoid moves)
            let mse_1day_val = f64::try_from(&mse_loss_1day).unwrap_or(f64::NAN);
            let mse_3day_val = f64::try_from(&mse_loss_3day).unwrap_or(f64::NAN);
            let dir_val = f64::try_from(&direction_loss).unwrap_or(f64::NAN);

            // Weighted combined loss (use shallow_clone to avoid moves)
            let mut loss = (mse_loss_1day.shallow_clone() * weight_1day_mse)
                + (mse_loss_3day.shallow_clone() * weight_3day_mse)
                + (direction_loss.shallow_clone() * weight_direction);

            // L2 regularization (weight decay) applied to all trainable parameters
            if weight_decay > 0.0 {
                // Sum squares of variables (if accessible)
                let mut l2_sum = Tensor::from(0.0).to_device(device);
                // VarStore::variables() returns a HashMap<String, Tensor>
                let vars_map = vs.variables();
                for (_name, v) in vars_map.iter() {
                    let s = v.pow_tensor_scalar(2).sum(tch::Kind::Float);
                    l2_sum = l2_sum + s;
                }
                // Use f64 to match tensor scalar conversion expectations
                let l2_term = l2_sum * (weight_decay as f64);
                loss = loss + l2_term;
            }

            // Debug: Check loss values
            let loss_val = f64::try_from(&loss).unwrap_or(f64::NAN);

            if !loss_val.is_finite() {
                eprintln!("⚠️  Warning: Loss is not finite: {}", loss_val);
                continue; // Skip backward pass
            }

            // Backward pass
            opt.zero_grad();
            loss.backward();
            opt.step();

            total_loss += loss_val;
            total_mse += (mse_1day_val * weight_1day_mse + mse_3day_val * weight_3day_mse);
            total_dir_loss += dir_val;
            num_batches += 1;
        }
    }

    Ok((
        total_loss / num_batches as f64,
        total_mse / num_batches as f64,
        total_dir_loss / num_batches as f64,
    ))
}

fn validate_epoch_stream(
    model: &TorchStockModel,
    datasets: &[DbStockDataset],
    batch_size: usize,
    device: Device,
    seq_len: usize,
) -> Result<f64> {
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    // Dual-task learning weights (same as training)
    let weight_1day_mse = 0.60;
    let weight_3day_mse = 0.25;
    let weight_direction = 0.15;

    // Reusable device buffers
    let mut dev_inputs_buf: Option<Tensor> = None;
    let mut dev_targets_buf: Option<Tensor> = None;

    for batch in StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len) {
        if let Ok((inputs_cpu, targets_cpu)) = batch {
            // Allocate or reuse device buffers with exact shapes
            let shape_inputs = inputs_cpu.size();
            let shape_targets = targets_cpu.size();
            if dev_inputs_buf.as_ref().map(|t| t.size()) != Some(shape_inputs.clone()) {
                dev_inputs_buf = Some(Tensor::zeros(&shape_inputs, (tch::Kind::Float, device)));
            }
            if dev_targets_buf.as_ref().map(|t| t.size()) != Some(shape_targets.clone()) {
                dev_targets_buf = Some(Tensor::zeros(&shape_targets, (tch::Kind::Float, device)));
            }

            let inputs = dev_inputs_buf.as_mut().unwrap();
            let targets = dev_targets_buf.as_mut().unwrap();
            inputs.copy_(&inputs_cpu);
            targets.copy_(&targets_cpu);

            // Forward pass with dual predictions
            let (pred_1day, pred_3day, _conf_1day, _conf_3day) = model.forward_dual(&inputs, false); // train=false for validation

            // Prepare targets (clone for separate use in both loss branches)
            let targets_1day = targets.shallow_clone();
            let targets_3day = targets.shallow_clone();

            // ========== 1-DAY LOSS ==========
            // Primary loss: MSE on 1-day predictions
            let mse_loss_1day = pred_1day.mse_loss(&targets_1day, tch::Reduction::Mean);

            // ========== 3-DAY LOSS (NEW) ==========
            // Secondary loss: MSE on 3-day predictions (same targets structure)
            let mse_loss_3day = pred_3day.mse_loss(&targets_3day, tch::Reduction::Mean);

            // ========== DIRECTION LOSS (COMBINED) ==========
            // Direction accuracy on price change (feature 12: close_pct)
            // close_pct already contains the direction information (percentage change from pre_close)
            let feature_index = 12i64;
            if pred_1day.size()[2] <= feature_index
                || pred_3day.size()[2] <= feature_index
                || targets.size()[2] <= feature_index
            {
                eprintln!(
                    "⚠️  Skipping batch: feature index {} out of bounds (pred_1day {:?}, pred_3day {:?}, targets {:?})",
                    feature_index,
                    pred_1day.size(),
                    pred_3day.size(),
                    targets.size()
                );
                continue;
            }
            let pred_pct_1day = pred_1day.i((.., .., feature_index)); // 1-day close_pct predictions
            let pred_pct_3day = pred_3day.i((.., .., feature_index)); // 3-day close_pct predictions
            let target_pct = targets.i((.., .., feature_index)); // actual close_pct

            // Direction: positive close_pct = up, negative = down
            let pred_direction_1day = pred_pct_1day.ge(0.0);
            let pred_direction_3day = pred_pct_3day.ge(0.0);
            let target_direction = target_pct.ge(0.0);

            // Direction loss: binary cross entropy on direction predictions (average of both)
            let direction_loss_1day = (pred_direction_1day.to_kind(tch::Kind::Float)
                - target_direction.to_kind(tch::Kind::Float))
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

            let direction_loss_3day = (pred_direction_3day.to_kind(tch::Kind::Float)
                - target_direction.to_kind(tch::Kind::Float))
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float);

            // Combined loss
            let direction_loss = (direction_loss_1day.shallow_clone() + direction_loss_3day) * 0.5;

            // ========== COMBINED WEIGHTED LOSS ==========
            // Multi-task learning: weighted combination of 1-day MSE, 3-day MSE, and direction loss
            let loss = (mse_loss_1day * weight_1day_mse)
                + (mse_loss_3day * weight_3day_mse)
                + (direction_loss * weight_direction);

            total_loss += f64::try_from(loss).unwrap_or(0.0);
            num_batches += 1;
        }
    }

    Ok(total_loss / num_batches as f64)
}

fn prepare_batch(
    batch: &[crate::dataset::StockItem],
    device: Device,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();

    // Build per-item normalized inputs/targets in parallel to reduce CPU time.
    let per_item: Vec<(Vec<f32>, Vec<f32>)> = batch
        .par_iter()
        .filter_map(|item| {
            if item.values.len() < seq_len + 1 {
                return None;
            }

            let reference_close_pct = item.values[seq_len - 1][12].abs().max(0.01);

            // Input: first seq_len timesteps (normalized)
            let mut local_inputs = Vec::with_capacity(seq_len * FEATURE_SIZE);
            for t in 0..seq_len {
                let normalized = normalize_features(item.values[t], reference_close_pct);
                if normalized.iter().any(|x: &f32| !x.is_finite()) {
                    eprintln!("⚠️  NaN in normalized input at timestep {} for stock", t);
                    return None;
                }
                local_inputs.extend_from_slice(&normalized);
            }

            // Target: last seq_len timesteps (shifted by 1, normalized)
            let mut local_targets = Vec::with_capacity(seq_len * FEATURE_SIZE);
            for t in 1..=seq_len {
                let normalized = normalize_features(item.values[t], reference_close_pct);
                if normalized.iter().any(|x: &f32| !x.is_finite()) {
                    eprintln!("⚠️  NaN in normalized target at timestep {} for stock", t);
                    return None;
                }
                local_targets.extend_from_slice(&normalized);
            }

            Some((local_inputs, local_targets))
        })
        .collect();

    // Flatten into contiguous buffers in original order.
    let mut input_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    let mut target_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    for (inputs_one, targets_one) in per_item.into_iter() {
        input_data.extend_from_slice(&inputs_one);
        target_data.extend_from_slice(&targets_one);
    }

    // If no valid items were prepared, skip this batch gracefully.
    let actual_items = input_data.len() / (seq_len * FEATURE_SIZE);
    if actual_items == 0 {
        anyhow::bail!("Empty prepared batch: no valid items");
    }

    // Create tensors and move once to device.
    let actual_batch = actual_items as i64;
    let inputs = Tensor::from_slice(&input_data)
        .reshape(&[actual_batch, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);
    let targets = Tensor::from_slice(&target_data)
        .reshape(&[actual_batch, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);

    Ok((inputs, targets))
}

struct StreamedBatches<'a> {
    datasets: &'a [DbStockDataset],
    batch_size: usize,
    device: Device,
    seq_len: usize,
    dataset_idx: usize,
    seq_idx: usize,
    buffer: Vec<crate::dataset::StockItem>,
}

impl<'a> StreamedBatches<'a> {
    fn new(
        datasets: &'a [DbStockDataset],
        batch_size: usize,
        device: Device,
        seq_len: usize,
    ) -> Self {
        Self {
            datasets,
            batch_size,
            device,
            seq_len,
            dataset_idx: 0,
            seq_idx: 0,
            buffer: Vec::with_capacity(batch_size),
        }
    }
}

impl<'a> Iterator for StreamedBatches<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.dataset_idx >= self.datasets.len() {
                if self.buffer.is_empty() {
                    return None;
                } else {
                    let batch = std::mem::take(&mut self.buffer);
                    return Some(prepare_batch(&batch, self.device, self.seq_len));
                }
            }

            let dataset = &self.datasets[self.dataset_idx];
            if self.seq_idx < dataset.len() {
                if let Some(item) = dataset.get(self.seq_idx) {
                    self.buffer.push(item);
                }
                self.seq_idx += 1;

                if self.buffer.len() == self.batch_size {
                    let batch = std::mem::take(&mut self.buffer);
                    return Some(prepare_batch(&batch, self.device, self.seq_len));
                }
            } else {
                self.dataset_idx += 1;
                self.seq_idx = 0;
            }
        }
    }
}

/// Parallel batch prefetch using Rayon thread pool
/// Pre-prepares multiple batches on CPU while GPU processes current batch
/// Returns iterator of prepared batches on device
fn prefetch_batches(
    datasets: &[DbStockDataset],
    total_batches: usize,
    batch_size: usize,
    device: Device,
    seq_len: usize,
    prefetch_depth: usize, // how many batches to prepare ahead
) -> Vec<Result<(Tensor, Tensor)>> {
    // Collect all raw batches first
    let raw_batches: Vec<_> = StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len)
        .take(total_batches)
        .filter_map(|res| res.ok())
        .collect();

    // Prepare in parallel using Rayon, then move to GPU
    raw_batches
        .into_par_iter()
        .map(|(inputs_cpu, targets_cpu)| {
            // Move to device: creates new tensors on target device
            let inputs_gpu = inputs_cpu.to(device);
            let targets_gpu = targets_cpu.to(device);
            Ok((inputs_gpu, targets_gpu))
        })
        .collect()
}
