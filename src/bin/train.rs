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
    let mut huber_delta: Option<f64> = None;
    let mut dropout_override: Option<f32> = None;

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
            "--dropout" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() {
                        dropout_override = Some(v);
                        i += 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    train_with_torch(
        train_records,
        val_records,
        device,
        learning_rate,
        out_dir,
        batch_size_override,
        weight_decay_override,
        huber_delta,
        dropout_override,
    )?;
    Ok(())
}
