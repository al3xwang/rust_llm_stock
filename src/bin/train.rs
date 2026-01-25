use rust_llm_stock::db::MlTrainingRecord;
use rust_llm_stock::training_torch::train_with_torch;
use std::env;
use std::error::Error;
use std::fs::File;
use tch::Device;

fn load_records_from_csv(path: &str) -> Result<Vec<MlTrainingRecord>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: MlTrainingRecord = result?;
        records.push(record);
    }
    Ok(records)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let mut train_path = "data/train.csv".to_string();
    let mut val_path = "data/val.csv".to_string();
    let mut device = Device::cuda_if_available();
    let mut learning_rate: Option<f64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                if i + 1 < args.len() {
                    train_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--val" => {
                if i + 1 < args.len() {
                    val_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--device" => {
                if i + 1 < args.len() {
                    let dev_str = args[i + 1].as_str();
                    if dev_str == "cpu" {
                        device = Device::Cpu;
                    } else if dev_str.starts_with("cuda") {
                        // Accept "cuda" or "cuda:<index>"
                        let parts: Vec<&str> = dev_str.split(':').collect();
                        let idx = if parts.len() > 1 {
                            parts[1].parse::<usize>().unwrap_or(0)
                        } else {
                            0usize
                        };
                        device = Device::Cuda(idx);
                    }
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        learning_rate = Some(v);
                        i += 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading training data from {}", train_path);
    let train_records = load_records_from_csv(&train_path)?;
    println!("Loading validation data from {}", val_path);
    let val_records = load_records_from_csv(&val_path)?;
    // Optional --out, --batch, --wd flags
    let mut i = 1; // restart arg parsing to pick up --out and other options
    let mut out_dir: Option<String> = None;
    let mut batch_size_override: Option<usize> = None;
    let mut weight_decay_override: Option<f64> = None;
    // Default to Huber loss with delta=1.0 (can be overridden with --huber-delta)
    let mut huber_delta: Option<f64> = Some(1.0);
    let mut grad_clip: Option<f64> = None;
    let mut compute_ic: bool = true;
    let mut topk_percentiles: Option<String> = None; // comma-separated list like "0.01,0.05"
    let mut lr_scheduler: Option<String> = None;
    let mut lr_min: Option<f64> = None;
    let mut cosine_t_max: Option<usize> = None;
    let mut dropout_override: Option<f32> = None;
    let mut max_epochs_override: Option<usize> = None;
    let mut early_stop_override: Option<usize> = None;

    // Direction weight (combined for both horizons) default 0.15
    let mut dir_weight: Option<f64> = None;

    // Sample-weighting & scheduler extras
    let mut t_mult: Option<f64> = None;
    let mut sample_weight_method: Option<String> = None;
    let mut sample_weight_decay: f64 = 0.0;
    let mut sample_weight_normalize: Option<String> = None;

    // Resume training from checkpoint
    let mut resume_from: Option<String> = None;

    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                if i + 1 < args.len() {
                    out_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--batch" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        batch_size_override = Some(v);
                        i += 1;
                    }
                }
            }
            "--wd" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        weight_decay_override = Some(v);
                        i += 1;
                    }
                }
            }
            "--huber-delta" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        huber_delta = Some(v);
                        i += 1;
                    }
                }
            }
            "--grad-clip" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        grad_clip = Some(v);
                        i += 1;
                    }
                }
            }
            "--no-ic" => {
                compute_ic = false;
            }
            "--topk-percentiles" => {
                if i + 1 < args.len() {
                    topk_percentiles = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--lr-scheduler" => {
                if i + 1 < args.len() {
                    lr_scheduler = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--lr-min" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        lr_min = Some(v);
                        i += 1;
                    }
                }
            }
            "--cosine-t-max" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        cosine_t_max = Some(v);
                        i += 1;
                    }
                }
            }
            "--t-mult" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        t_mult = Some(v);
                        i += 1;
                    }
                }
            }
            "--sample-weight-method" => {
                if i + 1 < args.len() {
                    sample_weight_method = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--sample-weight-decay" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        sample_weight_decay = v;
                        i += 1;
                    }
                }
            }
            "--sample-weight-normalize" => {
                if i + 1 < args.len() {
                    sample_weight_normalize = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--dropout" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        dropout_override = Some(v);
                        i += 1;
                    }
                }
            }
            "--max-epochs" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        max_epochs_override = Some(v);
                        i += 1;
                    }
                }
            }
            "--early-stop" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() {
                        early_stop_override = Some(v);
                        i += 1;
                    }
                }
            }
            "--dir-weight" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f64>() {
                        dir_weight = Some(v);
                        i += 1;
                    }
                }
            }
            "--resume-from" => {
                if i + 1 < args.len() {
                    resume_from = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Parse topk_percentiles into Vec<f64>
    let topk_vec: Vec<f64> = topk_percentiles
        .as_deref()
        .unwrap_or("0.01,0.05,0.10")
        .split(',')
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    train_with_torch(
        train_records,
        val_records,
        device,
        learning_rate,
        out_dir,
        batch_size_override,
        weight_decay_override,
        huber_delta,
        grad_clip,
        compute_ic,
        topk_vec,
        lr_scheduler,
        lr_min,
        cosine_t_max,
        t_mult,
        sample_weight_method,
        sample_weight_decay,
        sample_weight_normalize,
        dropout_override,
        max_epochs_override,
        early_stop_override,
        dir_weight,
        resume_from,
    )?;
    Ok(())
}
