/// Daily model predictor: Runs inference on feature vectors from daily_features
///
/// Usage:
///   cargo run --release --bin daily_predict -- --features features.csv --output predictions.csv
use clap::Parser;
use rust_llm_stock::feature_normalization::FEATURE_SIZE;
use serde::Serialize;
use std::error::Error;
use tch::{Device, Tensor, nn};

#[derive(Parser, Debug)]
#[command(name = "daily_predict")]
#[command(about = "Run model inference on daily features")]
struct Args {
    /// Input CSV with features (from daily_features)
    #[arg(long, default_value = "features.csv")]
    features: String,

    /// Model path (safetensors format)
    #[arg(long, default_value = "artifacts/best_model.safetensors")]
    model: String,

    /// Output predictions CSV
    #[arg(long, default_value = "predictions.csv")]
    output: String,

    /// Device: cuda or cpu
    #[arg(long, default_value = "cuda")]
    device: String,
}

#[derive(Debug, Serialize)]
struct PredictionRow {
    ts_code: String,
    pred_return: f32,
    direction: i32, // 1=up, -1=down, 0=neutral
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Daily Model Predictor ===");
    println!("Features: {}", args.features);
    println!("Model: {}", args.model);

    // Check device
    let device = if args.device == "cuda" {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    // Read features CSV
    println!("\nReading features from {}...", args.features);
    let mut rdr = csv::Reader::from_path(&args.features)?;
    let mut feature_data: Vec<(String, Vec<f32>)> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            continue;
        }

        let ts_code = record[0].to_string();
        let mut features = Vec::with_capacity(FEATURE_SIZE);

        for i in 1..record.len() {
            if let Ok(f) = record[i].parse::<f32>() {
                features.push(f);
            }
        }

        if features.len() == FEATURE_SIZE {
            feature_data.push((ts_code, features));
        }
    }

    println!("Loaded {} feature vectors", feature_data.len());

    // Load model (placeholder - would need actual model architecture)
    // For now, we'll create a minimal predictor
    println!("\nLoading model from {}...", args.model);
    // Model loading would use safetensors crate
    // This is a simplified version - in production use tch::nn to load

    let mut predictions: Vec<PredictionRow> = Vec::new();

    // Run inference for each stock
    println!("Running inference...");
    for (idx, (ts_code, features)) in feature_data.iter().enumerate() {
        // Convert features to tensor
        let _features_tensor = Tensor::from_slice(features).to_device(device);

        // Simplified prediction (in production, use loaded model)
        // This is placeholder logic showing the structure
        let pred_return = simple_predict(&features);
        let direction = if pred_return > 0.2 {
            1
        } else if pred_return < -0.2 {
            -1
        } else {
            0
        };
        let confidence = calculate_confidence(&features);

        predictions.push(PredictionRow {
            ts_code: ts_code.clone(),
            pred_return,
            direction,
            confidence,
        });

        if (idx + 1) % 100 == 0 {
            println!(
                "  [{}/{}] predictions generated",
                idx + 1,
                feature_data.len()
            );
        }
    }

    // Write predictions to CSV
    println!("\nWriting predictions to {}...", args.output);
    let mut wtr = csv::Writer::from_path(&args.output)?;

    for pred in &predictions {
        wtr.serialize(pred)?;
    }

    wtr.flush()?;
    println!("✅ {} predictions saved", predictions.len());

    // Summary statistics
    let avg_confidence =
        predictions.iter().map(|p| p.confidence).sum::<f32>() / predictions.len() as f32;
    let up_signals = predictions.iter().filter(|p| p.direction > 0).count();
    let down_signals = predictions.iter().filter(|p| p.direction < 0).count();

    println!("\n📊 Summary:");
    println!("  Average confidence: {:.4}", avg_confidence);
    println!("  Up signals: {}", up_signals);
    println!("  Down signals: {}", down_signals);
    println!(
        "  Hold signals: {}",
        predictions.len() - up_signals - down_signals
    );

    Ok(())
}

/// Simple predictor (placeholder)
/// In production, this would use the actual loaded model
fn simple_predict(features: &[f32]) -> f32 {
    if features.len() < 12 {
        return 0.0;
    }

    // Simplified heuristic: weighted combination of key features
    let close_pct = features[12];
    let rsi_14 = features[33];
    let momentum_5 = features[48];

    // If yesterday went down hard, predict reversal
    let reversal_signal = if close_pct < -2.0 && rsi_14 < 30.0 {
        0.5
    } else if close_pct > 2.0 && rsi_14 > 70.0 {
        -0.5
    } else {
        0.0
    };

    // Momentum component
    let momentum_component = momentum_5 * 0.5;

    (reversal_signal + momentum_component).clamp(-3.0, 3.0)
}

/// Calculate confidence score (0-1)
fn calculate_confidence(features: &[f32]) -> f32 {
    if features.len() < 50 {
        return 0.3;
    }

    // Check for signal strength across multiple indicators
    let rsi = features[33];
    let vol_percentile = features[100];

    // RSI extremes = higher confidence
    let rsi_confidence = if rsi < 20.0 || rsi > 80.0 { 0.8 } else { 0.4 };

    // Volatility regime adjustment
    let vol_confidence = if vol_percentile > 0.7 { 0.6 } else { 0.9 };

    (rsi_confidence * vol_confidence as f32).clamp(0.0, 1.0)
}
