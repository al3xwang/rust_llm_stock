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
    grad_clip: Option<f64>,
    compute_ic: bool,
    topk_percentiles: Vec<f64>,
    lr_scheduler: Option<String>,
    lr_min: Option<f64>,
    cosine_t_max: Option<usize>,
    t_mult: Option<f64>,
    sample_weight_method: Option<String>,
    sample_weight_decay: f64,
    sample_weight_normalize: Option<String>,
    dropout_override: Option<f32>,
    max_epochs_override: Option<usize>,
    early_stop_override: Option<usize>,
) -> Result<()> {
    println!("Initializing PyTorch model on {:?}...", device);

    // Config - Optimized for 80%+ GPU utilization
    // Desired sequence length (can be reduced automatically if dataset shorter)
    let desired_seq_len = 60; // ~3 months of context, aligns with longer-horizon features
    // Determine max possible seq_len from training records to avoid empty datasets
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for r in &train_records {
        *counts.entry(r.ts_code.clone()).or_insert(0) += 1;
    }
    let min_count = counts.values().min().cloned().unwrap_or(0);
    if min_count <= 1 {
        anyhow::bail!("Not enough records per-stock to build sequences (min_count={}).", min_count);
    }
    let seq_len = std::cmp::min(desired_seq_len, min_count - 1);
    // Defaults (can be overridden via CLI)
    let default_batch_size = 256; // Large batches to maximize GPU utilization
    let batch_size = batch_override.unwrap_or(default_batch_size);
    let default_max_epochs = 1000;
    let max_epochs = max_epochs_override.unwrap_or(default_max_epochs);

    // Scale learning rate with batch size: LR_new = LR_old * sqrt(batch_size_new / batch_size_old)
    let default_learning_rate = 1e-4 * (batch_size as f64 / 48.0_f64).sqrt();
    let learning_rate = override_lr.unwrap_or(default_learning_rate);

    let default_early_stop_patience = 20; // More patience for finding minima
    let early_stop_patience = early_stop_override.unwrap_or(default_early_stop_patience);
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
    println!("LR scheduler: {:?}, lr_min: {:?}, cosine_t_max: {:?}, t_mult: {:?}", lr_scheduler, lr_min, cosine_t_max, t_mult);
    println!("Sample weighting: method={:?} decay={} normalize={:?}", sample_weight_method, sample_weight_decay, sample_weight_normalize);
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
    // Cosine restart tracking
    // Ensure cycle length is at least 1 to avoid underflow when max_epochs <= warmup_epochs
    let mut cycle_t_max = cosine_t_max
        .map(|v| v as f64)
        .unwrap_or((max_epochs.saturating_sub(warmup_epochs) as f64).max(1.0));
    let mut t_in_cycle = 0.0f64;
    let t_mult_val = t_mult.unwrap_or(2.0);

    for epoch in 1..=max_epochs {
        println!(
            "\nEpoch {} (early stopping patience: {}/{})",
            epoch, epochs_without_improvement, early_stop_patience
        );

        // Learning rate warmup
        if epoch <= warmup_epochs {
            let warmup_lr = learning_rate * (epoch as f64 / warmup_epochs as f64);
            opt.set_lr(warmup_lr);
            current_lr = warmup_lr;
            if epoch == warmup_epochs {
                // Ensure optimizer and current_lr explicitly reflect the full learning rate
                opt.set_lr(learning_rate);
                current_lr = learning_rate;
                println!("  🔥 Warmup complete — using full learning rate {:.6}", current_lr);
            }
        } else if let Some(ref sched) = lr_scheduler {
            let s = sched.to_lowercase();
            // Cosine annealing scheduler: uses time since warmup
            if s == "cosine" {
                let t = (epoch - warmup_epochs) as f64; // 1..T
                let t_max = cosine_t_max.unwrap_or((max_epochs - warmup_epochs) as usize) as f64;
                let lr_min_eff = lr_min.unwrap_or(1e-6_f64);
                let new_lr = cosine_annealing_lr(learning_rate, lr_min_eff, t, t_max);
                opt.set_lr(new_lr);
                current_lr = new_lr;
                if (epoch - warmup_epochs) == 1 {
                    println!("  🔁 Using cosine annealing schedule (T_max={}), lr_min={:.6}", t_max as usize, lr_min_eff);
                }
            } else if s == "cosine_restart" {
                // SGDR-style cosine with restarts
                if epoch > warmup_epochs {
                    t_in_cycle += 1.0;
                    let lr_min_eff = lr_min.unwrap_or(1e-6_f64);
                    let new_lr = cosine_annealing_lr(learning_rate, lr_min_eff, t_in_cycle, cycle_t_max);
                    opt.set_lr(new_lr);
                    current_lr = new_lr;

                    if (epoch - warmup_epochs) == 1 {
                        println!("  🔁 Using cosine annealing with restarts (T0={}, t_mult={})", cycle_t_max as usize, t_mult_val);
                    }

                    if (t_in_cycle >= cycle_t_max) {
                        // schedule restart (next epoch will start new cycle)
                        t_in_cycle = 0.0;
                        cycle_t_max *= t_mult_val;
                        println!("  🔄 Restarting cosine schedule: new T_max={}", cycle_t_max as usize);
                    }
                }
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
            huber_delta,
            grad_clip,
            sample_weight_method.as_deref().unwrap_or("none"),
            sample_weight_decay,
            sample_weight_normalize.as_deref().unwrap_or("mean"),
        )?;
        println!(
            "  Train Loss: {:.6} (MSE: {:.6}, Dir: {:.6})",
            train_loss, train_mse, train_dir_loss
        );

        // Validation
        let valid_loss =
            validate_epoch_stream(&model, &valid_datasets, batch_size, device, seq_len, huber_delta, compute_ic, &topk_percentiles, sample_weight_method.as_deref().unwrap_or("none"), sample_weight_decay, sample_weight_normalize.as_deref().unwrap_or("mean"))?; // validation excludes weight decay term
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

            // Learning rate decay if plateauing (disabled when using cosine schedulers)
            if !lr_scheduler.as_deref().map(|s| s.to_lowercase().starts_with("cosine")).unwrap_or(false) {
                if epochs_since_lr_decay >= lr_patience {
                    current_lr *= lr_decay_factor;
                    opt.set_lr(current_lr);
                    epochs_since_lr_decay = 0;
                    println!("  📉 Reducing learning rate to {:.6}", current_lr);
                }
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
    huber_delta: Option<f64>,
    grad_clip: Option<f64>,
    sample_weight_method: &str,
    sample_weight_decay: f64,
    sample_weight_normalize: &str,
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
    let mut dev_weights_buf: Option<Tensor> = None;

    for batch in StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len, sample_weight_method, sample_weight_decay, sample_weight_normalize) {
        if let Ok((inputs_cpu, targets_cpu, weights_cpu, next_day_cpu)) = batch {
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

            // Prepare weights buffer (1D: batch)
            let shape_weights = vec![shape_inputs[0]];
            if dev_weights_buf.as_ref().map(|t| t.size()) != Some(shape_weights.clone()) {
                dev_weights_buf = Some(Tensor::zeros(&shape_weights, (tch::Kind::Float, device)));
            }

            let inputs = dev_inputs_buf.as_mut().unwrap();
            let targets = dev_targets_buf.as_mut().unwrap();
            let weights = dev_weights_buf.as_mut().unwrap();

            // Copy CPU → GPU without reallocating device tensors
            inputs.copy_(&inputs_cpu);
            targets.copy_(&targets_cpu);
            weights.copy_(&weights_cpu);

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
            // 1-day loss (weighted MSE or weighted Huber if specified)
            let mse_loss_1day = if let Some(delta) = huber_delta {
                weighted_huber_loss(&pred_1day, &targets_1day, &weights, delta)
            } else {
                weighted_mse_loss(&pred_1day, &targets_1day, &weights)
            };

            // 3-day loss (weighted MSE or weighted Huber if specified)
            let mse_loss_3day = if let Some(delta) = huber_delta {
                weighted_huber_loss(&pred_3day, &targets_3day, &weights, delta)
            } else {
                weighted_mse_loss(&pred_3day, &targets_3day, &weights)
            };

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

            // Gradient clipping (global norm) if requested
            if let Some(clip_val) = grad_clip {
                // Collect gradients from variable store
                let vars_map = vs.variables();
                let mut grads: Vec<Tensor> = Vec::new();
                for (_name, v) in vars_map.iter() {
                    // v.grad() returns the gradient tensor; may be zero if none
                    let g = v.grad();
                    // Skip if grad is undefined or empty
                    if g.numel() > 0 {
                        grads.push(g);
                    }
                }
                if !grads.is_empty() {
                    let total_norm = compute_global_grad_norm(&grads);
                    if total_norm.is_finite() && total_norm > (clip_val as f64) {
                        let scale = (clip_val as f64) / (total_norm + 1e-6);
                        scale_gradients_inplace(&grads, scale as f32);
                    }
                }
            }

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
    huber_delta: Option<f64>,
    compute_ic: bool,
    topk_percentiles: &[f64],
    sample_weight_method: &str,
    sample_weight_decay: f64,
    sample_weight_normalize: &str,
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
    let mut dev_weights_buf: Option<Tensor> = None;

    // Collect close_pct predictions and targets for IC/TopK metrics
    let mut preds_all: Vec<f64> = Vec::new();
    let mut targets_all: Vec<f64> = Vec::new();

    for batch in StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len, sample_weight_method, sample_weight_decay, sample_weight_normalize) {
        if let Ok((inputs_cpu, targets_cpu, weights_cpu, next_day_cpu)) = batch {
            // Allocate or reuse device buffers with exact shapes
            let shape_inputs = inputs_cpu.size();
            let shape_targets = targets_cpu.size();
            if dev_inputs_buf.as_ref().map(|t| t.size()) != Some(shape_inputs.clone()) {
                dev_inputs_buf = Some(Tensor::zeros(&shape_inputs, (tch::Kind::Float, device)));
            }
            if dev_targets_buf.as_ref().map(|t| t.size()) != Some(shape_targets.clone()) {
                dev_targets_buf = Some(Tensor::zeros(&shape_targets, (tch::Kind::Float, device)));
            }
            let shape_weights = vec![shape_inputs[0]];
            if dev_weights_buf.as_ref().map(|t| t.size()) != Some(shape_weights.clone()) {
                dev_weights_buf = Some(Tensor::zeros(&shape_weights, (tch::Kind::Float, device)));
            }

            let inputs = dev_inputs_buf.as_mut().unwrap();
            let targets = dev_targets_buf.as_mut().unwrap();
            let weights = dev_weights_buf.as_mut().unwrap();
            inputs.copy_(&inputs_cpu);
            targets.copy_(&targets_cpu);
            weights.copy_(&weights_cpu);

            // Forward pass with dual predictions
            let (pred_1day, pred_3day, _conf_1day, _conf_3day) = model.forward_dual(&inputs, false); // train=false for validation

            // Prepare targets (clone for separate use in both loss branches)
            let targets_1day = targets.shallow_clone();
            let targets_3day = targets.shallow_clone();

            // ========== 1-DAY LOSS ==========
            // Primary loss: MSE or Huber on 1-day predictions
            let mse_loss_1day = if let Some(delta) = huber_delta {
                weighted_huber_loss(&pred_1day, &targets_1day, &weights, delta)
            } else {
                weighted_mse_loss(&pred_1day, &targets_1day, &weights)
            };

            // ========== 3-DAY LOSS (NEW) ==========
            // Secondary loss: MSE or Huber on 3-day predictions (same targets structure)
            let mse_loss_3day = if let Some(delta) = huber_delta {
                weighted_huber_loss(&pred_3day, &targets_3day, &weights, delta)
            } else {
                weighted_mse_loss(&pred_3day, &targets_3day, &weights)
            };

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

            // Collect for IC/TopK computation (move to CPU and flatten)
            if compute_ic {
                let pred_cpu = pred_pct_1day.to_device(Device::Cpu);
                let next_cpu = next_day_cpu.to_device(Device::Cpu);
                let s0 = pred_cpu.size();
                if s0.len() == 2 {
                    let b = s0[0] as i64;
                    let s = s0[1] as i64;
                    for ii in 0..b {
                        for jj in 0..s {
                            let pv = pred_cpu.double_value(&[ii, jj]);
                            let tv = next_cpu.double_value(&[ii, jj]);
                            // skip NaN targets (missing next_day_return)
                            if tv.is_nan() {
                                continue;
                            }
                            preds_all.push(pv);
                            targets_all.push(tv);
                        }
                    }
                }
            }

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

    let avg_loss = total_loss / num_batches as f64;

    // Compute IC (Pearson + Spearman) and TopK metrics if requested
    if compute_ic && !preds_all.is_empty() {
        if let Some(pear) = pearson_corr(&preds_all, &targets_all) {
            let spearman = spearman_corr(&preds_all, &targets_all).unwrap_or(f64::NAN);
            println!("  IC Pearson: {:.4} Spearman: {:.4}", pear, spearman);

            let topk_stats = compute_topk_stats(&preds_all, &targets_all, topk_percentiles);
            let mut parts: Vec<String> = Vec::new();
            for (p, acc, mean_ret) in topk_stats {
                parts.push(format!("Top{:.2}% acc={:.3} mean={:.4}", p * 100.0, acc, mean_ret));
            }
            println!("  {}", parts.join(" | "));
        }
    }

    Ok(avg_loss)
}

    fn prepare_batch(
    batch: &[crate::dataset::StockItem],
    device: Device,
    seq_len: usize,
    sample_weight_method: &str,
    sample_weight_decay: f64,
    sample_weight_normalize: &str,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let batch_size = batch.len();

    // Build per-item normalized inputs/targets in parallel to reduce CPU time.
    let per_item: Vec<(Vec<f32>, Vec<f32>, Option<String>, Option<String>, Vec<Option<f32>>)> = batch
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

            Some((local_inputs, local_targets, item.last_trade_date.clone(), item.dataset_last_date.clone(), item.next_day_returns.clone()))
        })
        .collect();

    // Flatten into contiguous buffers in original order.
    let mut input_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    let mut target_data = Vec::with_capacity(batch_size * seq_len * FEATURE_SIZE);
    let mut next_day_data: Vec<f64> = Vec::with_capacity(batch_size * seq_len);
    let mut last_trade_dates: Vec<Option<String>> = Vec::with_capacity(batch_size);
    let mut dataset_last_dates: Vec<Option<String>> = Vec::with_capacity(batch_size);
    for (inputs_one, targets_one, last_dt, dataset_dt, nd_returns) in per_item.into_iter() {
        input_data.extend_from_slice(&inputs_one);
        target_data.extend_from_slice(&targets_one);
        last_trade_dates.push(last_dt);
        dataset_last_dates.push(dataset_dt);
        // nd_returns contains next_day_return per original values entry.
        // Targets correspond to nd_returns[1..=seq_len]
        for t in 1..=seq_len {
            let v = nd_returns.get(t).and_then(|o| *o).map(|x| x as f64).unwrap_or(f64::NAN);
            next_day_data.push(v);
        }
    }

    // If no valid items were prepared, skip this batch gracefully.
    let actual_items = input_data.len() / (seq_len * FEATURE_SIZE);
    if actual_items == 0 {
        anyhow::bail!("Empty prepared batch: no valid items");
    }

    // Compute sample weights (time-decay default)
    let mut weights: Vec<f32> = Vec::with_capacity(actual_items);
    for i in 0..actual_items {
        let w = if sample_weight_method == "time_decay" {
            if let (Some(ds), Some(last)) = (&dataset_last_dates[i], &last_trade_dates[i]) {
                if let (Ok(d0), Ok(d1)) = (
                    chrono::NaiveDate::parse_from_str(ds, "%Y%m%d"),
                    chrono::NaiveDate::parse_from_str(last, "%Y%m%d"),
                ) {
                    let age = (d0 - d1).num_days() as f64;
                    (-(sample_weight_decay * age)).exp() as f32
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };
        weights.push(w);
    }

    // Normalize weights
    if sample_weight_normalize == "mean" {
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        if mean != 0.0 {
            for v in weights.iter_mut() {
                *v /= mean;
            }
        }
    } else if sample_weight_normalize == "sum" {
        let sum: f32 = weights.iter().sum::<f32>();
        if sum != 0.0 {
            for v in weights.iter_mut() {
                *v /= sum;
            }
        }
    }

    // Create tensors and move once to device.
    let actual_batch = actual_items as i64;
    let inputs = Tensor::from_slice(&input_data)
        .reshape(&[actual_batch, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);
    let targets = Tensor::from_slice(&target_data)
        .reshape(&[actual_batch, seq_len as i64, FEATURE_SIZE as i64])
        .to_device(device);
    let weights_t = Tensor::from_slice(&weights).reshape(&[actual_batch]).to_device(device);

    // next_day_data length = actual_batch * seq_len
    let next_day_tensor = Tensor::from_slice(&next_day_data).reshape(&[actual_batch, seq_len as i64]).to_device(device);
    Ok((inputs, targets, weights_t, next_day_tensor))
}

struct StreamedBatches<'a> {
    datasets: &'a [DbStockDataset],
    batch_size: usize,
    device: Device,
    seq_len: usize,
    dataset_idx: usize,
    seq_idx: usize,
    buffer: Vec<crate::dataset::StockItem>,
    // sample weight config
    sample_weight_method: String,
    sample_weight_decay: f64,
    sample_weight_normalize: String,
}

impl<'a> StreamedBatches<'a> {
    fn new(
        datasets: &'a [DbStockDataset],
        batch_size: usize,
        device: Device,
        seq_len: usize,
        sample_weight_method: &str,
        sample_weight_decay: f64,
        sample_weight_normalize: &str,
    ) -> Self {
        Self {
            datasets,
            batch_size,
            device,
            seq_len,
            dataset_idx: 0,
            seq_idx: 0,
            buffer: Vec::with_capacity(batch_size),
            sample_weight_method: sample_weight_method.to_string(),
            sample_weight_decay,
            sample_weight_normalize: sample_weight_normalize.to_string(),
        }
    }
}

impl<'a> Iterator for StreamedBatches<'a> {
    type Item = Result<(Tensor, Tensor, Tensor, Tensor)>; // inputs, targets, weights, next_day_returns

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.dataset_idx >= self.datasets.len() {
                if self.buffer.is_empty() {
                    return None;
                } else {
                    let batch = std::mem::take(&mut self.buffer);
                    return Some(prepare_batch(&batch, self.device, self.seq_len, &self.sample_weight_method, self.sample_weight_decay, &self.sample_weight_normalize));
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
                    return Some(prepare_batch(&batch, self.device, self.seq_len, &self.sample_weight_method, self.sample_weight_decay, &self.sample_weight_normalize));
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
)-> Vec<(Tensor, Tensor, Tensor, Tensor)> {
    // Collect all raw batches first (use default sample-weighting params)
    let raw_batches: Vec<_> = StreamedBatches::new(datasets, batch_size, Device::Cpu, seq_len, "none", 0.0, "mean")
        .take(total_batches)
        .filter_map(|res| res.ok())
        .collect();

    // Prepare in parallel using Rayon, then move to GPU
    raw_batches
        .into_par_iter()
        .map(|(inputs_cpu, targets_cpu, weights_cpu, next_day_cpu)| {
            // Move to device: creates new tensors on target device
            let inputs_gpu = inputs_cpu.to(device);
            let targets_gpu = targets_cpu.to(device);
            let weights_gpu = weights_cpu.to(device);
            let next_gpu = next_day_cpu.to(device);
            (inputs_gpu, targets_gpu, weights_gpu, next_gpu)
        })
        .collect()
}

/// Huber loss implementation (returns mean over all elements)
pub(crate) fn huber_loss(pred: &Tensor, target: &Tensor, delta: f64) -> Tensor {
    // pred and target are expected to be same shape
    let diff = pred - target;
    let abs = diff.abs();
    let delta_t = Tensor::from(delta as f32).to_device(pred.device());
    // Elementwise comparison tensor <= tensor
    let mask = abs.le_tensor(&delta_t);

    let squared = diff.pow_tensor_scalar(2) * 0.5; // 0.5 * (diff^2)
    let linear = (&abs - (&delta_t * 0.5)) * &delta_t; // delta * (|diff| - 0.5*delta)

    let mask_f = mask.to_kind(tch::Kind::Float);
    let inv_mask_f = Tensor::from(1.0) - &mask_f;
    let huber = (mask_f * squared) + (inv_mask_f * linear);
    huber.mean(tch::Kind::Float)
}

/// Compute global L2 norm of gradients (grads are tensors representing gradient values)
pub(crate) fn compute_global_grad_norm(grads: &[Tensor]) -> f64 {
    let mut sum_sq = Tensor::from(0.0).to_device(grads[0].device());
    for g in grads.iter() {
        let s = g.pow_tensor_scalar(2).sum(tch::Kind::Float);
        sum_sq = sum_sq + s;
    }
    let norm = sum_sq.sqrt();
    f64::try_from(&norm).unwrap_or(f64::NAN)
}

/// Scale gradients in-place by a scalar factor
pub(crate) fn scale_gradients_inplace(grads: &[Tensor], scale: f32) {
    if (scale - 1.0).abs() < f32::EPSILON {
        return;
    }
    let s = Tensor::from(scale);
    for g in grads.iter() {
        // Multiply gradient by scale and copy back into same storage
        let scaled = g * &s;
        let mut target = g.shallow_clone();
        target.copy_(&scaled);
    }
}

/// Compute Pearson correlation between two slices (returns None if invalid)
pub(crate) fn pearson_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return None;
    }
    let n_f = n as f64;
    let mean_x = x.iter().sum::<f64>() / n_f;
    let mean_y = y.iter().sum::<f64>() / n_f;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return Some(0.0);
    }
    Some(cov / var_x.sqrt() / var_y.sqrt())
}

/// Compute Spearman correlation by ranking values then computing Pearson on ranks
pub(crate) fn spearman_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() == 0 || x.len() != y.len() {
        return None;
    }

    let rx = ranks(x);
    let ry = ranks(y);
    pearson_corr(&rx, &ry)
}

/// Produce simple ranks for values. Ranks are 1..n as f64.
fn ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0f64; n];
    for (pos, &i) in idxs.iter().enumerate() {
        ranks[i] = (pos + 1) as f64;
    }
    ranks
}

/// Compute TopK stats: for each percentile p (0<p<=1), compute directional accuracy and mean actual return among top-k by |pred|
pub(crate) fn compute_topk_stats(preds: &[f64], targets: &[f64], percentiles: &[f64]) -> Vec<(f64, f64, f64)> {
    let n = preds.len();
    if n == 0 || n != targets.len() {
        return Vec::new();
    }
    // Indices sorted by absolute predicted magnitude desc
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&a, &b| preds[b].abs().partial_cmp(&preds[a].abs()).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = Vec::new();
    for &p in percentiles.iter() {
        if !(p > 0.0 && p <= 1.0) {
            continue;
        }
        let k = ((n as f64) * p).floor() as usize;
        let k = std::cmp::max(1, k);
        let mut correct = 0usize;
        let mut sumret = 0.0;
        for &idx in idxs.iter().take(k) {
            let pred_sign = if preds[idx] >= 0.0 { 1.0 } else { -1.0 };
            let target_sign = if targets[idx] >= 0.0 { 1.0 } else { -1.0 };
            if pred_sign == target_sign {
                correct += 1;
            }
            sumret += targets[idx];
        }
        let acc = (correct as f64) / (k as f64);
        let meanret = sumret / (k as f64);
        results.push((p, acc, meanret));
    }
    results
}

/// Compute weighted MSE across batch where `weights` is 1D [batch] already normalized (mean=1 or other)
pub(crate) fn weighted_mse_loss(pred: &Tensor, target: &Tensor, weights: &Tensor) -> Tensor {
    // pred/target shape: [batch, seq, features]
    let diff = pred - target;
    let sq = diff.pow_tensor_scalar(2);
    // per-sample mean over remaining dims
    let sz = sq.size();
    let batch = sz[0];
    let _n_elems_per = (sz[1] * sz[2]) as f64; // previously unused; prefixed to avoid warning
    let per_sample = sq.reshape(&[batch, -1]).mean_dim(1, false, tch::Kind::Float);
    // weights shape [batch]
    let weighted = per_sample * weights;
    // sum weighted and divide by batch to keep comparable scale
    let s = weighted.sum(tch::Kind::Float);
    s / Tensor::from(batch as f64)
}

/// Compute weighted Huber loss per sample and aggregate similarly
pub(crate) fn weighted_huber_loss(pred: &Tensor, target: &Tensor, weights: &Tensor, delta: f64) -> Tensor {
    let diff = pred - target;
    let abs = diff.abs();
    let delta_t = Tensor::from(delta as f32).to_device(pred.device());

    let mask = abs.le_tensor(&delta_t);
    let squared = diff.pow_tensor_scalar(2) * 0.5;
    let linear = (&abs - (&delta_t * 0.5)) * &delta_t;
    let mask_f = mask.to_kind(tch::Kind::Float);
    let inv_mask_f = Tensor::from(1.0) - &mask_f;
    let huber_el = (mask_f * squared) + (inv_mask_f * linear);

    // per-sample mean
    let sz = huber_el.size();
    let batch = sz[0];
    let per_sample = huber_el.reshape(&[batch, -1]).mean_dim(1, false, tch::Kind::Float);
    let weighted = per_sample * weights;
    let s = weighted.sum(tch::Kind::Float);
    s / Tensor::from(batch as f64)
}

/// Cosine annealing LR schedule (t in [0,T])
pub(crate) fn cosine_annealing_lr(initial_lr: f64, lr_min: f64, t: f64, t_max: f64) -> f64 {
    if t_max <= 0.0 {
        return initial_lr;
    }
    let cos_val = (std::f64::consts::PI * (t / t_max)).cos();
    lr_min + 0.5 * (initial_lr - lr_min) * (1.0 + cos_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_huber_loss_small_and_large() {
        // Two elements: diff=0.5 (small), diff=2.0 (large)
        let pred = Tensor::from_slice(&[0.5f32, 2.0f32]).reshape(&[2, 1, 1]).to_device(Device::Cpu);
        let target = Tensor::zeros(&[2, 1, 1], (tch::Kind::Float, Device::Cpu));

        // delta = 1.0
        let loss = huber_loss(&pred, &target, 1.0);
        let got = f64::try_from(&loss).unwrap();

        // Expected: for 0.5 -> 0.5*(0.5^2)=0.125, for 2.0 -> 1*(2.0 - 0.5)=1.5 => mean=(0.125+1.5)/2
        let expected = (0.125 + 1.5) / 2.0;
        assert!((got - expected).abs() < 1e-6, "huber got {} expected {}", got, expected);
    }

    #[test]
    fn test_compute_and_scale_grad_norm() {
        // grads: [3,4] and [0,0] -> global norm = 5
        let g1 = Tensor::from_slice(&[3.0f32, 4.0f32]).to_device(Device::Cpu);
        let g2 = Tensor::from_slice(&[0.0f32, 0.0f32]).to_device(Device::Cpu);
        let grads = vec![g1.shallow_clone(), g2.shallow_clone()];

        let norm = compute_global_grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-6, "norm {} != 5", norm);

        // scale to max_norm 2.5 -> factor 0.5
        scale_gradients_inplace(&grads, 0.5);
        let new_norm = compute_global_grad_norm(&grads);
        assert!((new_norm - 2.5).abs() < 1e-6, "new_norm {} != 2.5", new_norm);
    }

    #[test]
    fn test_grad_clip_applies_to_varstore_grads() {
        // Create a small varstore and a single parameter
        let mut vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let w = root.var("w", &[2], tch::nn::Init::Const(1.0));

        // Create a synthetic loss that produces gradient [3.0,4.0] for w
        // Loss = w[0]*3 + w[1]*4 => gradient wrt w is [3,4]
        let three = Tensor::from(3.0f32);
        let four = Tensor::from(4.0f32);
        let loss = w.i((0,)) * &three + w.i((1,)) * &four;
        let loss_sum = loss.sum(tch::Kind::Float);

        loss_sum.backward();

        // Collect gradients from varstore
        let vars_map = vs.variables();
        let mut grads: Vec<Tensor> = Vec::new();
        for (_name, v) in vars_map.iter() {
            let g = v.grad();
            if g.numel() > 0 {
                grads.push(g);
            }
        }

        let norm = compute_global_grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-6, "expected norm 5 got {}", norm);

        // Apply clipping to 2.5
        scale_gradients_inplace(&grads, 0.5);
        let new_norm = compute_global_grad_norm(&grads);
        assert!((new_norm - 2.5).abs() < 1e-6, "expected new norm 2.5 got {}", new_norm);
    }

    #[test]
    fn test_ic_and_topk_helpers() {
        // Small synthetic vectors
        let preds = vec![0.1, -0.2, 0.05, 0.5, -0.3];
        let targets = vec![0.08, -0.25, 0.02, 0.6, -0.1];

        // Pearson should be positive
        let pear = pearson_corr(&preds, &targets).unwrap();
        assert!(pear > 0.9);

        // Spearman should also be positive
        let sp = spearman_corr(&preds, &targets).unwrap();
        assert!(sp >= 0.9 - 1e-9);

        // TopK: top 20% -> top 1 element (n=5 -> k=1) which is pred 0.5 -> target 0.6 -> acc=1.0 mean=0.6
        let stats = compute_topk_stats(&preds, &targets, &[0.2]);
        assert_eq!(stats.len(), 1);
        let (p, acc, meanret) = stats[0];
        assert!((p - 0.2).abs() < 1e-9);
        assert!((acc - 1.0).abs() < 1e-9);
        assert!((meanret - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let lr0 = 1.0;
        let lr_min = 0.0;
        // t=0 -> lr=lr0
        let v0 = cosine_annealing_lr(lr0, lr_min, 0.0, 10.0);
        assert!((v0 - lr0).abs() < 1e-9);
        // t=T -> lr=lr_min
        let vT = cosine_annealing_lr(lr0, lr_min, 10.0, 10.0);
        assert!((vT - lr_min).abs() < 1e-9);
        // midpoint t=T/2 -> lr ~ (lr0+lr_min)/2
        let vm = cosine_annealing_lr(lr0, lr_min, 5.0, 10.0);
        assert!((vm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_restarts() {
        let lr0 = 1.0;
        let lr_min = 0.0;
        let mut cycle_t_max = 3.0;
        let t_mult_val = 2.0;
        let mut t_in_cycle = 0.0;
        let mut seen_restart = false;

        // simulate 5 epochs after warmup
        for epoch in 1..=5 {
            t_in_cycle += 1.0;
            let lr = cosine_annealing_lr(lr0, lr_min, t_in_cycle, cycle_t_max);
            if (epoch as f64) == cycle_t_max {
                // at cycle boundary lr should be near lr_min
                assert!((lr - lr_min).abs() < 1e-9);
                // trigger restart
                t_in_cycle = 0.0;
                cycle_t_max *= t_mult_val;
                seen_restart = true;
            }
        }
        assert!(seen_restart);
        assert!((cycle_t_max - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_mse_and_huber_uniform_weights() {
        // Two samples, each with single scalar feature
        let pred = Tensor::from_slice(&[0.5f32, 2.0f32]).reshape(&[2, 1, 1]).to_device(Device::Cpu);
        let target = Tensor::zeros(&[2, 1, 1], (tch::Kind::Float, Device::Cpu));
        let weights = Tensor::from_slice(&[1.0f32, 1.0f32]).reshape(&[2]).to_device(Device::Cpu);

        // MSE per-sample: 0.25 and 4.0 -> per-sample means are same; weighted_mse should equal mean of per-sample means
        let mse_w = f64::try_from(&weighted_mse_loss(&pred, &target, &weights)).unwrap();
        let expected_mse = (0.25 + 4.0) / 2.0;
        assert!((mse_w - expected_mse).abs() < 1e-6);

        // Huber with delta=1.0: first -> 0.5*0.25=0.125; second -> linear part: 1*(2.0 - 0.5)=1.5; mean=(0.125+1.5)/2
        let huber_w = f64::try_from(&weighted_huber_loss(&pred, &target, &weights, 1.0)).unwrap();
        let expected_huber = (0.125 + 1.5) / 2.0;
        assert!((huber_w - expected_huber).abs() < 1e-6);
    }

    #[test]
    fn test_training_step_with_sample_weights() {
        // Use small seq_len; DummyStockDataset produces values length = seq_len + 1 so we pass seq_len here
        let seq_len = 3usize;
        let ds = crate::dataset::DummyStockDataset::new(2, seq_len + 1);
        let mut item1 = ds.get(0).unwrap();
        let mut item2 = ds.get(1).unwrap();

        // Set metadata so time-decay weights are different
        item1.last_trade_date = Some("20260120".to_string());
        item1.dataset_last_date = Some("20260122".to_string());
        item2.last_trade_date = Some("20260121".to_string());
        item2.dataset_last_date = Some("20260122".to_string());

        // Prepare batch with time-decay weighting and mean normalization
        let (inputs, targets, weights) = prepare_batch(&[item1, item2], Device::Cpu, seq_len, "time_decay", 0.1, "mean").unwrap();

        // Verify computed normalized weights approximately match expectation
        let w0 = weights.double_value(&[0]);
        let w1 = weights.double_value(&[1]);
        let raw0 = (-(0.1_f64 * 2.0)).exp();
        let raw1 = (-(0.1_f64 * 1.0)).exp();
        let mean = (raw0 + raw1) / 2.0;
        let exp0 = raw0 / mean;
        let exp1 = raw1 / mean;
        assert!((w0 - exp0).abs() < 1e-6, "weights mismatch: {} vs {}", w0, exp0);
        assert!((w1 - exp1).abs() < 1e-6, "weights mismatch: {} vs {}", w1, exp1);

        // Build a tiny model and run a single training step using the weighted loss
        let mut vs = nn::VarStore::new(Device::Cpu);
        let model = TorchStockModel::new(&vs.root(), &ModelConfig::default());
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        opt.set_lr(1e-3);
        let (pred_1day, _pred_3day, _c1, _c3) = model.forward_dual(&inputs, true);
        let loss = weighted_mse_loss(&pred_1day, &targets, &weights);
        let loss_val = f64::try_from(&loss).unwrap();
        assert!(loss_val.is_finite());

        opt.zero_grad();
        loss.backward();
        opt.step();
    }}



