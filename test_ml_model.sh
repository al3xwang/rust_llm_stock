#!/bin/bash

# Quick ML Model Test - Verify your trained model works
# Usage: ./test_ml_model.sh

set -e

echo "═══════════════════════════════════════════════════════════"
echo "🧪 Testing Your Trained ML Model"
echo "═══════════════════════════════════════════════════════════"
echo

# Step 1: Check model file
echo "Step 1: Checking model file..."
if [ -f "artifacts/best_model.safetensors" ]; then
    MODEL_SIZE=$(ls -lh artifacts/best_model.safetensors | awk '{print $5}')
    MODEL_DATE=$(ls -l artifacts/best_model.safetensors | awk '{print $6, $7, $8}')
    echo "  ✅ Model found: $MODEL_SIZE (created: $MODEL_DATE)"
else
    echo "  ❌ Model not found!"
    exit 1
fi
echo

# Step 2: Check PyTorch
echo "Step 2: Checking PyTorch library..."
if [ -d "libtorch" ]; then
    echo "  ✅ LibTorch installed"
    ls -d libtorch/lib/*.dylib 2>/dev/null | wc -l | xargs echo "  Libraries found:"
else
    echo "  ❌ LibTorch not found!"
    exit 1
fi
echo

# Step 3: Check database connection
echo "Step 3: Checking database..."
DB_URL="${DATABASE_URL:-postgresql://postgres:12341234@localhost:5432/research}"
if psql "$DB_URL" -c "SELECT COUNT(*) FROM ml_training_dataset LIMIT 1" >/dev/null 2>&1; then
    TOTAL_RECORDS=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM ml_training_dataset")
    echo "  ✅ Database connected"
    echo "  Records in ml_training_dataset: $TOTAL_RECORDS"
else
    echo "  ❌ Database connection failed!"
    exit 1
fi
echo

# Step 4: Test prediction on small sample
echo "Step 4: Running test predictions (5 stocks)..."
echo "  This may take 30-60 seconds..."
echo

cargo run --release --features pytorch --bin batch_predict -- \
    --model-path artifacts/best_model.safetensors \
    --limit 5 \
    --min-confidence 0.0 2>&1 | grep -E "(Processing|Predicted|Saving|✓)" | tail -20

echo
echo "═══════════════════════════════════════════════════════════"
echo "✅ Model Test Complete!"
echo "═══════════════════════════════════════════════════════════"
echo
echo "Your model is ready for daily trading!"
echo
echo "Next steps:"
echo "  1. Generate today's predictions:"
echo "     ./daily_ml_trading.sh"
echo
echo "  2. Or check recent predictions:"
echo "     psql \$DATABASE_URL -c \"SELECT * FROM stock_predictions ORDER BY prediction_date DESC LIMIT 10\""
echo
echo "  3. Read the full guide:"
echo "     cat ML_TRADING_GUIDE.md"
echo
