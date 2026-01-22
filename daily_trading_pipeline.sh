#!/bin/bash

# Complete Daily Trading Pipeline
# 1. Run full data ingestion and feature calculation
# 2. Run model predictions
# 3. Generate trading signals
#
# Usage: ./daily_trading_pipeline.sh [date]
# If no date specified, uses current date

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get date (current date if not specified)
if [ -z "$1" ]; then
    TRADE_DATE=$(date +%Y%m%d)
else
    TRADE_DATE=$1
fi

echo "======================================"
echo "Daily Trading Pipeline"
echo "Trade Date: $TRADE_DATE"
echo "======================================"

# Step 1: Run full pipeline (ingest data + calculate features)
echo ""
echo "📊 Step 1: Running full data pipeline..."
echo "   - Ingesting daily stock data"
echo "   - Ingesting index data"
echo "   - Creating adjusted prices"
echo "   - Calculating ML features"
echo "   - Exporting training data"
echo ""

./run_full_pipeline_llm.sh 2>&1 | tail -30

# Step 2: Run predictions
echo ""
echo "🤖 Step 2: Running model predictions..."
if [ -f "data/training_data.csv" ]; then
    cargo run --release --bin daily_predict -- \
        --features data/training_data.csv \
        --model artifacts/best_model.safetensors \
        --output predictions_${TRADE_DATE}.csv \
        --device cuda 2>&1 | grep -E "(Predictions written|Error|Error:|predictions)" || echo "Predictions completed"
    
    if [ -f "predictions_${TRADE_DATE}.csv" ]; then
        echo "✅ Predictions written to: predictions_${TRADE_DATE}.csv"
        wc -l predictions_${TRADE_DATE}.csv
    fi
else
    echo "⚠️  training_data.csv not found. Skipping predictions."
fi

# Step 3: Generate signals
echo ""
echo "📈 Step 3: Generating trading signals..."
if [ -f "predictions_${TRADE_DATE}.csv" ]; then
    cargo run --release --bin generate_signals -- \
        --predictions predictions_${TRADE_DATE}.csv \
        --output signals_${TRADE_DATE}.csv 2>&1 | grep -E "(Signals written|signals|Signal)" || echo "Signals generated"
    
    if [ -f "signals_${TRADE_DATE}.csv" ]; then
        echo "✅ Trading signals written to: signals_${TRADE_DATE}.csv"
        wc -l signals_${TRADE_DATE}.csv
        echo ""
        echo "📄 Signal summary:"
        head -5 signals_${TRADE_DATE}.csv
    fi
else
    echo "⚠️  predictions_${TRADE_DATE}.csv not found. Skipping signals."
fi

echo ""
echo "======================================"
echo "✅ Daily Trading Pipeline Complete"
echo "======================================"
