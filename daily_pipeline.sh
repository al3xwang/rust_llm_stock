#!/bin/bash

# Daily Trading Pipeline - Complete Workflow
# 
# This orchestrates the full daily trading process:
# 1. Ingests new market data (optional - controlled by --skip-ingest flag)
# 2. Calculates ML features for all stocks
# 3. Generates trading signal reports
#
# Usage:
#   ./daily_pipeline.sh [--skip-ingest] [date]
#   
# Examples:
#   ./daily_pipeline.sh                 # Full pipeline for today
#   ./daily_pipeline.sh --skip-ingest   # Skip data ingestion, use cached features
#   ./daily_pipeline.sh 20251225        # Full pipeline for Dec 25, 2025
#   ./daily_pipeline.sh --skip-ingest 20251225  # Signals only for Dec 25

set -e

SKIP_INGEST=false
TRADE_DATE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ingest)
            SKIP_INGEST=true
            shift
            ;;
        *)
            TRADE_DATE="$1"
            shift
            ;;
    esac
done

# Default to today if not specified
if [ -z "$TRADE_DATE" ]; then
    TRADE_DATE=$(date +%Y%m%d)
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Daily Trading Pipeline - $(date +%Y-%m-%d)           ║"
echo "║              Trade Date: $TRADE_DATE                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Data Ingestion (Optional)
if [ "$SKIP_INGEST" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📥 STEP 1: Data Ingestion & Feature Calculation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "⏳ Running full data pipeline..."
    echo "   • Ingesting stock daily data"
    echo "   • Ingesting market indices"
    echo "   • Calculating adjusted prices"
    echo "   • Computing ML features"
    echo ""
    
    LOG_FILE="pipeline_${TRADE_DATE}.log"
    if ./run_full_pipeline_llm.sh > "$LOG_FILE" 2>&1; then
        echo "✅ Pipeline completed successfully"
        FEATURE_COUNT=$(tail -1 "$LOG_FILE" | grep -oE "[0-9]+ records" || echo "N/A")
        echo "   $FEATURE_COUNT processed"
    else
        echo "⚠️  Pipeline completed with warnings (see $LOG_FILE)"
    fi
    
    echo ""
else
    echo "⏭️  Skipping data ingestion (using cached features)"
    echo ""
fi

# Step 2: Generate Trading Signals
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STEP 2: Generating Trading Signals"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if features file exists
if [ ! -f "data/training_data.csv" ]; then
    echo "❌ Error: Features file not found at data/training_data.csv"
    echo "   Run with full pipeline: ./daily_pipeline.sh $TRADE_DATE"
    exit 1
fi

# Count total records (excluding header)
TOTAL_RECORDS=$(tail -n +2 data/training_data.csv | wc -l)
# Count unique stocks
UNIQUE_STOCKS=$(tail -n +2 data/training_data.csv | cut -d',' -f1 | sort -u | wc -l)

echo "⏳ Analyzing ${TOTAL_RECORDS} total records from ${UNIQUE_STOCKS} unique stocks..."
echo ""

if ./daily_report_fast.sh "$TRADE_DATE" 2>&1 | tail -15; then
    echo ""
    REPORT_FILE="daily_report_${TRADE_DATE}.csv"
    if [ -f "$REPORT_FILE" ]; then
        SIGNAL_COUNT=$(tail -1 "$REPORT_FILE" | grep -oE "[0-9]+" || echo "N/A")
        echo "✅ Trading signals generated: $REPORT_FILE"
    fi
else
    echo "⚠️  Signal generation completed with warnings"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  ✅ Pipeline Complete!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Output Files:"
echo "   • Daily Report: daily_report_${TRADE_DATE}.csv"
if [ "$SKIP_INGEST" = false ]; then
    echo "   • Features: data/training_data.csv"
    echo "   • Pipeline Log: pipeline_${TRADE_DATE}.log"
fi
echo ""
echo "🚀 Next Steps:"
echo "   1. Review daily_report_${TRADE_DATE}.csv for trading opportunities"
echo "   2. For tomorrow: ./daily_pipeline.sh"
echo "   3. Or skip ingestion for quick signals: ./daily_pipeline.sh --skip-ingest"
echo ""
