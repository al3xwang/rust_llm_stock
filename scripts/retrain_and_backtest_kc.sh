#!/bin/bash
# Retrain KC stacked model and run comprehensive backtest before production deployment
# Usage: ./scripts/retrain_and_backtest_kc.sh

set -e

echo "========================================================================"
echo "🚀 KC MODEL RETRAINING & BACKTESTING PIPELINE"
echo "========================================================================"
echo ""
echo "📊 This pipeline will:"
echo "   1. Verify training data availability"
echo "   2. Train stacked ensemble model (LightGBM + XGBoost + MLP)"
echo "   3. Generate test predictions with probability calibration"
echo "   4. Run comprehensive backtest analysis"
echo "   5. Generate performance reports and visualizations"
echo ""

# Set working directory
cd "$(dirname "$0")/.."
WORK_DIR=$(pwd)
echo "📁 Working directory: $WORK_DIR"

# Activate virtual environment if it exists
if [ -d "./venv" ]; then
    echo "🐍 Activating Python virtual environment..."
    source ./venv/bin/activate
fi
echo ""

# Check if data files exist
echo "========================================================================"
echo "Step 1: Verifying Training Data"
echo "========================================================================"

if [ ! -f "./data/kc_train.csv" ]; then
    echo "❌ ERROR: kc_train.csv not found!"
    echo "   Please run the export pipeline first:"
    echo "   cargo run --bin export_training_data --release -- \\"
    echo "     --start-date 20220101 --end-date 20251231 --market-type KC \\"
    echo "     --listed-before 20231231 --train-val-test-split \"0.6 0.2 0.2\" \\"
    echo "     --train-output ./data/kc_train.csv \\"
    echo "     --val-output ./data/kc_val.csv \\"
    echo "     --test-holdout-output ./data/kc_test.csv"
    exit 1
fi

if [ ! -f "./data/kc_val.csv" ]; then
    echo "❌ ERROR: kc_val.csv not found!"
    exit 1
fi

if [ ! -f "./data/kc_test.csv" ]; then
    echo "❌ ERROR: kc_test.csv not found!"
    exit 1
fi

echo "✅ Training data verified:"
echo "   📂 kc_train.csv: $(wc -l < ./data/kc_train.csv | xargs) rows"
echo "   📂 kc_val.csv:   $(wc -l < ./data/kc_val.csv | xargs) rows"
echo "   📂 kc_test.csv:  $(wc -l < ./data/kc_test.csv | xargs) rows"
echo ""

# Create artifacts directory
mkdir -p artifacts

# Step 2: Train the stacked ensemble model
echo "========================================================================"
echo "Step 2: Training Stacked Ensemble Model"
echo "========================================================================"
echo "🤖 Training models with upsampling and class-weight tuning..."
echo "   - LightGBM base models with various scale_pos_weights"
echo "   - XGBoost models with sample weights"
echo "   - MLP (Neural Network) classifiers"
echo "   - Probability calibration on validation set"
echo "   - Logistic regression stacker"
echo ""

python3 py/train_kc_sampling_and_stack.py

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi

echo ""
echo "✅ Model training completed successfully"
echo "   📦 Models saved in ./artifacts/"
echo ""

# Check if model artifacts were created
if [ ! -f "./artifacts/stacking_summary_kc.json" ]; then
    echo "❌ ERROR: stacking_summary_kc.json not found after training!"
    exit 1
fi

# Display training summary
echo "========================================================================"
echo "📋 Training Summary"
echo "========================================================================"
cat artifacts/stacking_summary_kc.json | python3 -m json.tool
echo ""

# Step 3: Generate test predictions
echo "========================================================================"
echo "Step 3: Generating Test Set Predictions"
echo "========================================================================"
echo "🔮 Applying trained model to test data..."
echo ""

# The training script already generates test predictions
if [ -f "./artifacts/test_predictions_stacked_kc.csv" ]; then
    echo "✅ Test predictions generated:"
    echo "   📄 test_predictions_stacked_kc.csv ($(wc -l < ./artifacts/test_predictions_stacked_kc.csv | xargs) rows)"
else
    echo "⚠️  Warning: test_predictions_stacked_kc.csv not found"
    echo "   Checking for alternative prediction files..."
    ls -lh artifacts/*pred*.csv 2>/dev/null || echo "   No prediction files found"
fi
echo ""

# Step 4: Run backtest
echo "========================================================================"
echo "Step 4: Running Comprehensive Backtest"
echo "========================================================================"
echo "📊 Evaluating model performance on test set..."
echo "   - Direction accuracy metrics"
echo "   - Returns analysis by prediction"
echo "   - Trading strategy simulation"
echo "   - Risk metrics and drawdown analysis"
echo "   - Performance visualizations"
echo ""

python3 py/backtest_kc_model.py

if [ $? -ne 0 ]; then
    echo "❌ Backtest failed!"
    exit 1
fi

echo ""
echo "✅ Backtest completed successfully"
echo ""

# Step 5: Generate final report
echo "========================================================================"
echo "Step 5: Final Evaluation Report"
echo "========================================================================"

# Check for generated plots
if [ -d "./artifacts/plots" ]; then
    PLOT_COUNT=$(ls -1 ./artifacts/plots/*.png 2>/dev/null | wc -l | xargs)
    echo "📈 Generated $PLOT_COUNT visualization plots in ./artifacts/plots/"
fi

# Show key metrics from training summary
echo ""
echo "🎯 Key Performance Metrics:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "./artifacts/stacking_summary_kc.json" ]; then
    python3 << 'EOF'
import json
with open('./artifacts/stacking_summary_kc.json') as f:
    summary = json.load(f)
    
print(f"   Training Accuracy:   {summary.get('train_acc', 0):.2%}")
print(f"   Validation Accuracy: {summary.get('val_acc', 0):.2%}")
print(f"   Test Accuracy:       {summary.get('test_acc', 0):.2%}")
print(f"   Test Precision:      {summary.get('test_precision', 0):.2%}")
print(f"   Test Recall:         {summary.get('test_recall', 0):.2%}")
print(f"   Test F1-Score:       {summary.get('test_f1', 0):.2%}")
print(f"   Test ROC-AUC:        {summary.get('test_auc', 0):.4f}")

if 'optimal_threshold' in summary:
    print(f"\n   Optimal Threshold:   {summary['optimal_threshold']:.4f}")
    print(f"   (Optimized on validation set)")
EOF
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Production readiness check
echo "========================================================================"
echo "🚦 Production Readiness Assessment"
echo "========================================================================"

python3 << 'EOF'
import json
import sys

with open('./artifacts/stacking_summary_kc.json') as f:
    summary = json.load(f)

test_acc = summary.get('test_acc', 0)
test_precision = summary.get('test_precision', 0)
test_recall = summary.get('test_recall', 0)
test_auc = summary.get('test_auc', 0)

# Production criteria
MIN_ACCURACY = 0.52
MIN_PRECISION = 0.50
MIN_AUC = 0.52

print("Checking production deployment criteria:")
print(f"   ✓ Test Accuracy >= {MIN_ACCURACY:.0%}:  {'✅ PASS' if test_acc >= MIN_ACCURACY else '❌ FAIL'} ({test_acc:.2%})")
print(f"   ✓ Test Precision >= {MIN_PRECISION:.0%}: {'✅ PASS' if test_precision >= MIN_PRECISION else '❌ FAIL'} ({test_precision:.2%})")
print(f"   ✓ Test ROC-AUC >= {MIN_AUC:.2f}:    {'✅ PASS' if test_auc >= MIN_AUC else '❌ FAIL'} ({test_auc:.4f})")
print("")

if test_acc >= MIN_ACCURACY and test_precision >= MIN_PRECISION and test_auc >= MIN_AUC:
    print("🎉 ✅ MODEL READY FOR PRODUCTION DEPLOYMENT")
    print("")
    print("Next steps:")
    print("   1. Review backtest results in ./artifacts/plots/")
    print("   2. Copy model artifacts to production:")
    print("      - artifacts/stacked_ensemble_kc.pkl")
    print("      - artifacts/stacking_summary_kc.json")
    print("   3. Deploy to prediction pipeline")
    sys.exit(0)
else:
    print("⚠️  MODEL NEEDS IMPROVEMENT BEFORE PRODUCTION")
    print("")
    print("Recommendations:")
    if test_acc < MIN_ACCURACY:
        print(f"   • Improve accuracy (current: {test_acc:.2%}, target: >={MIN_ACCURACY:.0%})")
    if test_precision < MIN_PRECISION:
        print(f"   • Improve precision (current: {test_precision:.2%}, target: >={MIN_PRECISION:.0%})")
    if test_auc < MIN_AUC:
        print(f"   • Improve AUC (current: {test_auc:.4f}, target: >={MIN_AUC:.2f})")
    print("   • Consider feature engineering improvements")
    print("   • Try different model configurations")
    print("   • Adjust class weights and thresholds")
    sys.exit(1)
EOF

READINESS_STATUS=$?

echo ""
echo "========================================================================"
echo "✨ PIPELINE COMPLETED"
echo "========================================================================"
echo ""
echo "📂 Outputs:"
echo "   - Model artifacts:      ./artifacts/"
echo "   - Training summary:     ./artifacts/stacking_summary_kc.json"
echo "   - Test predictions:     ./artifacts/test_predictions_stacked_kc.csv"
echo "   - Backtest plots:       ./artifacts/plots/"
echo ""

exit $READINESS_STATUS
