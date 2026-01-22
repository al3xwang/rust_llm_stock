#!/bin/bash

set -e

echo "=== Daily Stock Data Update Workflow ==="
echo ""

# Load environment variables
export $(cat .env | xargs)

# Step 1: Pull latest daily stock data
echo "Step 1: Pulling latest daily stock data..."
echo "----------------------------------------"
../target/release/pullall-daily
echo ""

# Step 2: Update adjusted_stock_daily for stocks with adjustment events
echo "Step 2: Updating adjusted prices for stocks with adjustment events..."
echo "---------------------------------------------------------------------"
# Use 7-day lookback by default (can be overridden with first argument)
LOOKBACK_DAYS=${1:-7}
../target/release/update_adjusted_daily $LOOKBACK_DAYS
echo ""

echo "=== Daily Update Complete ==="
echo ""
echo "Summary:"
echo "  ✓ Latest daily data pulled from Tushare"
echo "  ✓ Adjusted prices updated for stocks with adjustment events (${LOOKBACK_DAYS}-day lookback)"
echo ""
echo "Next steps:"
echo "  - Run dataset_creator to refresh training datasets if needed"
