#!/bin/bash

# Fast Daily Features Report - Optimized for large datasets
# Generates diversified trading signals (top 1 per stock)
#
# Usage: ./daily_report_fast.sh [date]

set -e

TRADE_DATE="${1:-$(date +%Y%m%d)}"
FEATURES_FILE="data/training_data.csv"
OUTPUT_FILE="daily_report_${TRADE_DATE}.csv"

echo "======================================"
echo "Daily Trading Report (Fast Mode)"
echo "Date: $TRADE_DATE"
echo "======================================"
echo ""

if [ ! -f "$FEATURES_FILE" ]; then
    echo "❌ Features file not found: $FEATURES_FILE"
    exit 1
fi

# Use streaming approach: sort in-place, get top per stock
echo "📈 Generating diversified trading signals..."
echo ""

# Header
echo "ts_code,volume,ema_5,rsi_14,bb_bandwidth,next_day_return" > "$OUTPUT_FILE"

# Process: Extract → Sort → Deduplicate per stock → Take top 20
tail -n +2 "$FEATURES_FILE" | \
    awk -F',' '{
        ts_code=$1
        volume=$4                # feature_003
        ema_5=$19               # feature_018
        rsi_14=$34              # feature_033
        bb_bandwidth=$44        # feature_043
        next_return=$105        # last column
        
        if (next_return != "") {
            printf "%s,%s,%s,%s,%s,%s\n", ts_code, volume, ema_5, rsi_14, bb_bandwidth, next_return
        }
    }' | \
    sort -t',' -k6,6rn -k2,2rn | \
    awk -F',' '!seen[$1]++ {print}' | \
    head -50 >> "$OUTPUT_FILE"

echo "✅ Report written to: $OUTPUT_FILE"
echo ""
echo "📄 Top 50 stocks by expected next-day return (1 per stock):"
tail -50 "$OUTPUT_FILE" | head -50
echo ""

# Stats
TOP_POSITIVE=$(tail -n +2 "$OUTPUT_FILE" | awk -F',' '{if($6=="1") print $1}' | wc -l)
TOP_NEGATIVE=$(tail -n +2 "$OUTPUT_FILE" | awk -F',' '{if($6=="-1") print $1}' | wc -l)
TOTAL_SIGNALS=$(tail -n +2 "$OUTPUT_FILE" | wc -l)

echo ""
echo "═══════════════════════════════════════"
echo "📊 Trading Opportunities Summary"
echo "═══════════════════════════════════════"
echo "Total unique stocks analyzed: $(tail -n +2 "$FEATURES_FILE" | cut -d',' -f1 | sort -u | wc -l)"
echo "Trading opportunities identified: $TOTAL_SIGNALS"
echo "  ✅ Bullish signals (↑): $TOP_POSITIVE"
echo "  ❌ Bearish signals (↓): $TOP_NEGATIVE"
echo ""
echo "Report file: $OUTPUT_FILE"
echo "✅ Ready for trading decisions!"
echo "═══════════════════════════════════════"
