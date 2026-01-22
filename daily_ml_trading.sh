#!/bin/bash

# Daily ML-Based Trading Report Generator
# Uses trained PyTorch model for stock predictions
# Usage: ./daily_ml_trading.sh [YYYYMMDD] [--top N]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="artifacts/best_model.safetensors"
TOP_N=50
TARGET_DATE="${1:-$(date +%Y%m%d)}"
MIN_CONFIDENCE=0.5  # Only show predictions with >50% confidence

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --top)
            TOP_N="$2"
            shift 2
            ;;
        --confidence)
            MIN_CONFIDENCE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            TARGET_DATE="$1"
            shift
            ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "📊 ML-Based Daily Trading Report"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $TARGET_DATE"
echo "Model: $MODEL_PATH"
echo "Top N: $TOP_N stocks"
echo "Min Confidence: $MIN_CONFIDENCE"
echo "═══════════════════════════════════════════════════════════"
echo

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Available models:"
    ls -lh artifacts/*.safetensors 2>/dev/null || echo "  No models found in artifacts/"
    exit 1
fi

# Check if PyTorch is available
if [ ! -d "libtorch" ]; then
    echo "⚠️  Warning: PyTorch (libtorch) not found!"
    echo "Falling back to feature-based trading report..."
    echo
    exec ./daily_report_fast.sh "$TARGET_DATE"
fi

# Step 1: Run batch predictions for target date
echo "Step 1/3: Running ML predictions..."
echo "─────────────────────────────────────────────────────────"

# Use batch_predict to generate predictions for specific date
cargo run --release --features pytorch --bin batch_predict -- \
    --model-path "$MODEL_PATH" \
    --min-confidence "$MIN_CONFIDENCE" 2>&1 | tail -20 &

PREDICT_PID=$!

# Wait for predictions with timeout
TIMEOUT=300  # 5 minutes
ELAPSED=0
while kill -0 $PREDICT_PID 2>/dev/null; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "⚠️  Prediction taking too long, killing process..."
        kill $PREDICT_PID 2>/dev/null || true
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo -n "."
done
echo
echo "✓ Predictions completed"
echo

# Step 2: Query predictions from database
echo "Step 2/3: Fetching top $TOP_N predictions..."
echo "─────────────────────────────────────────────────────────"

OUTPUT_FILE="ml_trading_report_${TARGET_DATE}.csv"

psql "${DATABASE_URL:-postgresql://postgres:12341234@localhost:5432/research}" > "$OUTPUT_FILE" <<EOF
\copy (
    SELECT 
        sp.ts_code,
        sb.name,
        sb.industry,
        sp.predicted_return,
        sp.confidence,
        CASE 
            WHEN sp.predicted_direction THEN 'BUY ↑'
            ELSE 'SELL ↓'
        END as signal,
        sp.trade_date,
        sp.model_version
    FROM stock_predictions sp
    LEFT JOIN stock_basic sb ON sp.ts_code = sb.ts_code
    WHERE sp.trade_date = '$TARGET_DATE'
        AND sp.confidence >= $MIN_CONFIDENCE
    ORDER BY 
        ABS(sp.predicted_return) DESC,
        sp.confidence DESC
    LIMIT $TOP_N
) TO STDOUT WITH CSV HEADER
EOF

if [ ! -s "$OUTPUT_FILE" ]; then
    echo "⚠️  No predictions found for $TARGET_DATE"
    echo "Possible reasons:"
    echo "  1. No data available for this date yet"
    echo "  2. Model hasn't generated predictions"
    echo "  3. Confidence threshold too high ($MIN_CONFIDENCE)"
    echo
    echo "Checking latest available predictions..."
    psql "${DATABASE_URL:-postgresql://postgres:12341234@localhost:5432/research}" <<EOF2
SELECT 
    MAX(trade_date) as latest_prediction_date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT ts_code) as unique_stocks,
    AVG(confidence) as avg_confidence
FROM stock_predictions;
EOF2
    exit 1
fi

echo "✓ Retrieved predictions"
echo

# Step 3: Display summary
echo "Step 3/3: Generating report..."
echo "─────────────────────────────────────────────────────────"

# Count signals
TOTAL_STOCKS=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
BUY_SIGNALS=$(tail -n +2 "$OUTPUT_FILE" | grep -c "BUY" || echo 0)
SELL_SIGNALS=$(tail -n +2 "$OUTPUT_FILE" | grep -c "SELL" || echo 0)

echo
echo "═══════════════════════════════════════════════════════════"
echo "📊 ML Trading Report Summary"
echo "═══════════════════════════════════════════════════════════"
echo "Date: $TARGET_DATE"
echo "Model: $(basename $MODEL_PATH)"
echo "─────────────────────────────────────────────────────────"
echo "Total signals: $TOTAL_STOCKS"
echo "  🟢 BUY signals:  $BUY_SIGNALS"
echo "  🔴 SELL signals: $SELL_SIGNALS"
echo "═══════════════════════════════════════════════════════════"
echo

# Display top 10 in terminal
echo "📈 Top 10 Trading Opportunities:"
echo "─────────────────────────────────────────────────────────"
head -11 "$OUTPUT_FILE" | column -t -s,
echo

echo "✅ Full report saved to: $OUTPUT_FILE"
echo
echo "Next steps:"
echo "  1. Review the signals: cat $OUTPUT_FILE"
echo "  2. Backtest: Compare with actual outcomes tomorrow"
echo "  3. Adjust confidence: Try --confidence 0.6 for higher quality"
echo

exit 0
