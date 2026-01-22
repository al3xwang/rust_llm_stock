#!/bin/bash

# Quick Trading Signal Pipeline
# Runs predictions and signal generation on existing training data
#
# Usage: ./quick_signals.sh [features_file] [date]

set -e

FEATURES_FILE="${1:-data/training_data.csv}"
TRADE_DATE="${2:-$(date +%Y%m%d)}"

echo "======================================"
echo "Quick Trading Signals Pipeline"
echo "Features: $FEATURES_FILE"
echo "Date: $TRADE_DATE"
echo "======================================"

if [ ! -f "$FEATURES_FILE" ]; then
    echo "❌ Features file not found: $FEATURES_FILE"
    exit 1
fi

FEATURE_COUNT=$(wc -l < "$FEATURES_FILE")
echo "✅ Found $FEATURE_COUNT records in features file"
echo ""

# Step 1: Run predictions
echo "🤖 Running model predictions..."
PRED_OUTPUT="predictions_${TRADE_DATE}.csv"

cargo run --release --features pytorch --bin daily_predict -- \
    --features "$FEATURES_FILE" \
    --model artifacts/best_model.safetensors \
    --output "$PRED_OUTPUT" \
    --device cuda 2>&1 | tail -20

if [ -f "$PRED_OUTPUT" ]; then
    PRED_COUNT=$(wc -l < "$PRED_OUTPUT")
    echo "✅ Predictions written: $PRED_OUTPUT ($PRED_COUNT records)"
    echo ""
fi

# Step 2: Generate trading signals
echo "📈 Generating trading signals..."
SIGNAL_OUTPUT="signals_${TRADE_DATE}.csv"

if [ -f "$PRED_OUTPUT" ]; then
    cargo run --release --bin generate_signals -- \
        --predictions "$PRED_OUTPUT" \
        --output "$SIGNAL_OUTPUT" 2>&1 | tail -20
    
    if [ -f "$SIGNAL_OUTPUT" ]; then
        SIGNAL_COUNT=$(wc -l < "$SIGNAL_OUTPUT")
        echo "✅ Signals written: $SIGNAL_OUTPUT ($SIGNAL_COUNT records)"
        echo ""
        echo "📄 Sample signals:"
        head -10 "$SIGNAL_OUTPUT"
    fi
fi

echo ""
echo "======================================"
echo "✅ Pipeline Complete!"
echo "======================================"
