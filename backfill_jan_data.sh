#!/usr/bin/env bash
# Quick script to backfill January 2026 data into ml_training_dataset
# Since dataset_creator processes incrementally, we'll force it through the holiday gap

set -e

cd /Users/alex/stock-analysis-workspace/rust_llm_stock

echo "=== Backfilling January 2026 Data ==="
echo ""

# Step 1: Delete records before Jan to allow reprocessing through the gap
echo "Step 1: Clearing dates from Dec 27-31 to force reprocessing..."
psql postgresql://postgres:12341234@127.0.0.1:5432/research <<'ENDSQL'
DELETE FROM ml_training_dataset WHERE trade_date >= '20251227';
SELECT 'Cleared Dec 27-31' as status;
ENDSQL

echo ""
echo "Step 2: Running dataset_creator (attempt 1 - process 20251227-20251231)..."
cargo run --release --bin dataset_creator 2>&1 | grep -E "(Incremental|Processed|inserted)"

echo ""
echo "Step 3: Running dataset_creator (attempt 2 - process 20260105-20260114)..."
cargo run --release --bin dataset_creator 2>&1 | grep -E "(Incremental|Processed|inserted)"

echo ""
echo "Step 4: Checking final data..."
psql postgresql://postgres:12341234@127.0.0.1:5432/research <<'ENDSQL'
SELECT 
    trade_date, 
    COUNT(*) as stock_count,
    MIN(next_day_return) as min_return,
    MAX(next_day_return) as max_return,
    AVG(next_day_return) as avg_return
FROM ml_training_dataset 
WHERE trade_date IN ('20260112', '20260113', '20260114')
GROUP BY trade_date
ORDER BY trade_date;

SELECT 
    'Overall stats' as metric,
    COUNT(*) as total_records,
    COUNT(DISTINCT trade_date) as distinct_dates,
    MIN(trade_date) as earliest,
    MAX(trade_date) as latest
FROM ml_training_dataset;
ENDSQL

echo ""
echo "=== Backfill Complete ==="
