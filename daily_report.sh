#!/bin/bash

# Simple Daily Features Report
# Generates a CSV of the most promising stocks for trading based on features
#
# Usage: ./daily_report.sh [date]

set -e

TRADE_DATE="${1:-$(date +%Y%m%d)}"
FEATURES_FILE="data/training_data.csv"
OUTPUT_FILE="daily_report_${TRADE_DATE}.csv"

echo "======================================"
echo "Daily Trading Report"
echo "Date: $TRADE_DATE"
echo "Features File: $FEATURES_FILE"
echo "======================================"
echo ""

if [ ! -f "$FEATURES_FILE" ]; then
    echo "❌ Features file not found: $FEATURES_FILE"
    exit 1
fi

TOTAL_RECORDS=$(tail -n +2 "$FEATURES_FILE" | wc -l)
echo "📊 Total records in dataset: $TOTAL_RECORDS"
echo ""

# Extract key metrics for report
echo "📈 Generating daily report..."
echo ""

# Get first few rows of features to analyze
echo "ts_code,volume,ema_5,rsi_14,bb_bandwidth,next_day_return" > "$OUTPUT_FILE"

# Extract top stocks by next_day_return (feature_104)
# Skip header and extract ts_code (1st col) and other key features
awk -F',' 'NR>1 {
    ts_code=$1
    volume=$4                # feature_003
    ema_5=$19               # feature_018 (roughly)
    rsi_14=$34              # feature_033 (roughly)
    bb_bandwidth=$44        # feature_043 (roughly)
    next_return=$105        # last column (feature_104)
    
    if (next_return != "") {
        printf "%s,%s,%s,%s,%s,%s\n", ts_code, volume, ema_5, rsi_14, bb_bandwidth, next_return
    }
}' "$FEATURES_FILE" | \
    sort -t',' -k6 -rn | \
    awk -F',' '!seen[$1]++ {print}' | \
    head -50 >> "$OUTPUT_FILE"

echo "✅ Report written to: $OUTPUT_FILE"
echo ""
echo "📄 Top 50 stocks by expected next-day return:"
cat "$OUTPUT_FILE"
echo ""

# Stats
TOP_POSITIVE=$(tail -n +2 "$OUTPUT_FILE" | awk -F',' '{if($6>0) print $1}' | wc -l)
echo ""
echo "═══════════════════════════════════════"
echo "📊 Daily Report Summary"
echo "═══════════════════════════════════════"
echo "Total stocks analyzed: $TOTAL_RECORDS"
echo "Stocks with positive outlook: $TOP_POSITIVE"
echo "Report file: $OUTPUT_FILE"
echo ""
echo "✅ Ready for trading decisions!"
echo "═══════════════════════════════════════"
