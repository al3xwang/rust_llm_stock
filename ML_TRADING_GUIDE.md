# ML-Based Daily Trading Guide

## Quick Start

You have **trained PyTorch models** ready for daily trading predictions!

### Option 1: Automated ML Trading (Recommended)
```bash
# Generate today's ML-based trading report (top 50 stocks)
./daily_ml_trading.sh

# Generate for specific date
./daily_ml_trading.sh 20251225

# Get top 100 stocks with high confidence
./daily_ml_trading.sh --top 100 --confidence 0.6
```

### Option 2: Manual Prediction Pipeline
```bash
# Step 1: Run batch predictions for all stocks
cargo run --release --features pytorch --bin batch_predict -- \
    --model-path artifacts/best_model.safetensors \
    --min-confidence 0.5

# Step 2: Query predictions from database
psql $DATABASE_URL -c "
SELECT ts_code, predicted_return, confidence, predicted_direction
FROM stock_predictions
WHERE trade_date = '20251225'
ORDER BY ABS(predicted_return) DESC
LIMIT 50;"
```

## Available Models

Check your trained models:
```bash
ls -lh artifacts/*.safetensors
```

Models you have:
- ✅ **best_model.safetensors** (37MB) - **Use this for production**
- Checkpoints: epoch_10, 20, 30, 40, 50, 60, 70, 80, 90

## How It Works

### 1. Model Architecture
- **Input**: 105 normalized features (OHLCV + technical indicators)
- **Output**: Predicted next-day return + confidence score
- **Training**: MSE loss + directional accuracy (15% weight)
- **Best validation loss**: ~0.004154

### 2. Prediction Flow
```
Stock Data (DB) → Feature Extraction → Normalization → Model Inference → Predictions (DB)
```

### 3. Output Format
```csv
ts_code,name,industry,predicted_return,confidence,signal,trade_date,model_version
000001.SZ,平安银行,银行,0.0234,0.87,BUY ↑,20251225,pytorch_v1.0
```

## Daily Trading Workflow

### Morning Routine (Before Market Open)
```bash
# 1. Generate ML predictions
./daily_ml_trading.sh

# 2. Review top opportunities
cat ml_trading_report_$(date +%Y%m%d).csv

# 3. Cross-check with feature-based signals
./daily_report_fast.sh

# 4. Execute trades based on high-confidence BUY signals
```

### End of Day (After Market Close)
```bash
# 1. Download latest data
cargo run --release --bin pullall-stock-daily

# 2. Update features
cargo run --release --bin dataset_creator

# 3. Export fresh training data
cargo run --release --bin export_training_data
```

## Command Reference

### Generate Predictions
```bash
# All stocks, default settings
./daily_ml_trading.sh

# Specific date
./daily_ml_trading.sh 20251225

# Top 100 with 60% min confidence
./daily_ml_trading.sh --top 100 --confidence 0.6

# Use specific model checkpoint
./daily_ml_trading.sh --model artifacts/checkpoint_epoch_90.safetensors
```

### Direct Binary Usage
```bash
# Batch predict all stocks
cargo run --release --features pytorch --bin batch_predict -- \
    --model-path artifacts/best_model.safetensors \
    --min-confidence 0.5

# Limit to 100 stocks (testing)
cargo run --release --features pytorch --bin batch_predict -- \
    --limit 100

# Use GPU (if available)
cargo run --release --features pytorch --bin batch_predict -- \
    --use-gpu
```

### Query Database Directly
```bash
# Get today's predictions
psql $DATABASE_URL <<EOF
SELECT ts_code, predicted_return, confidence, 
       CASE WHEN predicted_direction THEN 'BUY' ELSE 'SELL' END as signal
FROM stock_predictions
WHERE trade_date = CURRENT_DATE::text
  AND confidence > 0.6
ORDER BY ABS(predicted_return) DESC
LIMIT 20;
EOF

# Check prediction accuracy (if actual data available)
psql $DATABASE_URL <<EOF
SELECT 
    COUNT(*) as total,
    AVG(CASE WHEN prediction_correct THEN 1.0 ELSE 0.0 END) as accuracy
FROM stock_predictions
WHERE actual_return IS NOT NULL;
EOF
```

## Interpreting Results

### Signal Strength
| Confidence | Meaning | Action |
|-----------|---------|--------|
| 0.8-1.0 | Very high | Strong conviction trade |
| 0.6-0.8 | High | Good opportunity |
| 0.5-0.6 | Medium | Moderate confidence |
| <0.5 | Low | Skip or use small position |

### Return Magnitude
| Predicted Return | Interpretation |
|-----------------|----------------|
| >3% | Strong bullish signal |
| 1-3% | Moderate upside |
| -1% to 1% | Neutral/noise |
| -3% to -1% | Moderate downside |
| <-3% | Strong bearish signal |

### Combined Strategy
Best signals: **High confidence (>0.6) + High absolute return (>2%)**

Example filter:
```bash
awk -F',' '$5 > 0.6 && ($4 > 0.02 || $4 < -0.02)' ml_trading_report_20251225.csv
```

## Troubleshooting

### "Model not found"
```bash
# Check available models
ls -lh artifacts/*.safetensors

# Verify path
export MODEL_PATH="artifacts/best_model.safetensors"
./daily_ml_trading.sh --model $MODEL_PATH
```

### "PyTorch not found"
Your system needs libtorch for model inference:
```bash
# Check if libtorch exists
ls -ld libtorch

# If missing, automatic fallback to feature-based reports
# Script will show: "⚠️ Warning: PyTorch not found! Falling back..."
```

### "No predictions for date"
```bash
# Check latest predictions
psql $DATABASE_URL -c "SELECT MAX(trade_date) FROM stock_predictions;"

# Generate predictions manually
cargo run --release --features pytorch --bin batch_predict
```

### Empty output
- Date may be in future (no data yet)
- Confidence threshold too high (try --confidence 0.3)
- Model hasn't run for this date yet

## Comparing ML vs Feature-Based

### Run both methods:
```bash
# ML-based (requires trained model)
./daily_ml_trading.sh > ml_signals.txt

# Feature-based (always works)
./daily_report_fast.sh > feature_signals.txt

# Compare
diff ml_signals.txt feature_signals.txt
```

### When to use each:

**ML Model (daily_ml_trading.sh)**
- ✅ Learned patterns from historical data
- ✅ Accounts for complex interactions
- ✅ Confidence scores for risk management
- ❌ Requires training updates
- ❌ Black box (harder to interpret)

**Feature-Based (daily_report_fast.sh)**
- ✅ Always available (no model needed)
- ✅ Transparent rules
- ✅ Fast (<15 seconds)
- ❌ Fixed strategy
- ❌ No confidence scores

**Best Practice**: Use both! ML for primary signals, features for validation.

## Backtesting Your Model

### Step 1: Generate historical predictions
```bash
# Predict for past month
for date in $(seq -f "%Y%m%d" 20241201 20241231); do
    ./daily_ml_trading.sh $date
done
```

### Step 2: Compare with actual returns
```bash
psql $DATABASE_URL <<EOF
SELECT 
    sp.trade_date,
    COUNT(*) as predictions,
    AVG(CASE WHEN sp.predicted_direction = (ar.actual_return > 0) 
        THEN 1.0 ELSE 0.0 END) as directional_accuracy,
    CORR(sp.predicted_return, ar.actual_return) as return_correlation
FROM stock_predictions sp
JOIN (
    SELECT ts_code, trade_date, 
           (close - LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)) / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) as actual_return
    FROM adjusted_stock_daily
) ar ON sp.ts_code = ar.ts_code AND sp.trade_date = ar.trade_date
WHERE sp.trade_date >= '20241201'
GROUP BY sp.trade_date
ORDER BY sp.trade_date;
EOF
```

## Retraining Your Model

When market conditions change:
```bash
# 1. Update data
./run_full_pipeline_llm.sh

# 2. Retrain
cargo run --release --features pytorch --bin train

# 3. Test new model
./daily_ml_trading.sh --model artifacts/best_model.safetensors
```

## Advanced Usage

### Parallel Processing (Large Datasets)
```bash
# Split by offset (useful for distributed systems)
cargo run --release --features pytorch --bin batch_predict -- --offset 0 --limit 100 &
cargo run --release --features pytorch --bin batch_predict -- --offset 100 --limit 100 &
wait
```

### Custom Model Versions
```bash
# Test different checkpoints
for epoch in 10 20 30 40 50 60 70 80 90; do
    echo "Testing epoch $epoch..."
    ./daily_ml_trading.sh --model artifacts/checkpoint_epoch_${epoch}.safetensors --top 10
done
```

### Export for External Systems
```bash
# Generate JSON format
psql $DATABASE_URL -t -A -F"," -c "
SELECT json_agg(row_to_json(t))
FROM (
    SELECT * FROM stock_predictions 
    WHERE trade_date = '20251225' 
    ORDER BY confidence DESC 
    LIMIT 50
) t" > predictions.json
```

## Next Steps

1. **Start using**: `./daily_ml_trading.sh` to get today's predictions
2. **Track performance**: Save each day's report and compare with actual outcomes
3. **Refine threshold**: Adjust `--confidence` based on accuracy observations
4. **Combine strategies**: Use ML + feature-based for higher conviction
5. **Automate**: Add to cron for automatic daily reports

---

**Need help?** Check the logs and compare with feature-based reports for validation.
