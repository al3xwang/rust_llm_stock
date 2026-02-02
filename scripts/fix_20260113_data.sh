#!/bin/bash

# Fix 20260113 data by deleting and re-inserting with proper historical context
# This resolves NULL values in technical indicators (rsi_14, bb_upper, atr) caused by
# insufficient historical data in the original insertion

set -e

echo "========================================="
echo "Fixing 20260113 data with historical context"
echo "========================================="

# Step 1: Delete existing 20260113 data
echo ""
echo "[Step 1] Deleting existing 20260113 data from ml_training_dataset..."
psql $DATABASE_URL <<EOF
DELETE FROM ml_training_dataset WHERE trade_date = '20260113';
SELECT COUNT(*) as remaining_records FROM ml_training_dataset WHERE trade_date = '20260113';
EOF

# Step 2: Re-run dataset_creator to process 20260113 with proper historical context
echo ""
echo "[Step 2] Re-running dataset_creator to process 20260113..."
echo "This will:"
echo "  - Fetch 400 days of historical data (back to ~20240721)"
echo "  - Calculate all technical indicators with full context"
echo "  - Insert 20260113 records with complete feature values"
echo ""

cd "$(dirname "$0")"
cargo run --release --bin dataset_creator

# Step 3: Verify fix
echo ""
echo "[Step 3] Verifying 20260113 data quality..."
psql $DATABASE_URL <<EOF
SELECT 
  COUNT(*) as total_records,
  COUNT(rsi_14) as rsi_count,
  COUNT(bb_upper) as bb_count,
  COUNT(atr) as atr_count,
  COUNT(pe_percentile_52w) as pe_count
FROM ml_training_dataset 
WHERE trade_date = '20260113';

-- Sample record inspection
SELECT 
  ts_code, 
  trade_date,
  COALESCE(rsi_14::text, 'NULL') as rsi_14,
  COALESCE(bb_upper::text, 'NULL') as bb_upper,
  COALESCE(atr::text, 'NULL') as atr,
  COALESCE(pe_percentile_52w::text, 'NULL') as pe_percentile_52w
FROM ml_training_dataset 
WHERE trade_date = '20260113'
LIMIT 1;
EOF

echo ""
echo "========================================="
echo "✅ Fix complete! 20260113 data re-inserted with full historical context"
echo "========================================="
