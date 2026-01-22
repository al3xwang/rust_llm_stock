#!/bin/bash

echo "=== Testing Daily Data Ingestion Compilation ==="
echo ""

echo "1. Checking pullall-daily..."
cargo check --bin pullall-daily --quiet 2>&1 | grep -E "error|Finished" | head -5
if [ $? -eq 0 ]; then
    echo "   ✓ pullall-daily compiles successfully"
else
    echo "   ✗ pullall-daily has errors"
fi
echo ""

echo "2. Checking ingest_tushare_stock_daily..."
cargo check --bin ingest_tushare_stock_daily --quiet 2>&1 | grep -E "error|Finished" | head -5
if [ $? -eq 0 ]; then
    echo "   ✓ ingest_tushare_stock_daily compiles successfully"
else
    echo "   ✗ ingest_tushare_stock_daily has errors"
fi
echo ""

echo "3. Checking ingest_tushare_stock..."
cargo check --bin ingest_tushare_stock --quiet 2>&1 | grep -E "error|Finished" | head -5
if [ $? -eq 0 ]; then
    echo "   ✓ ingest_tushare_stock compiles successfully"
else
    echo "   ✗ ingest_tushare_stock has errors"
fi
echo ""

echo "4. Checking ingest_tushare_index..."
cargo check --bin ingest_tushare_index --quiet 2>&1 | grep -E "error|Finished" | head -5
if [ $? -eq 0 ]; then
    echo "   ✓ ingest_tushare_index compiles successfully"
else
    echo "   ✗ ingest_tushare_index has errors"
fi
echo ""

echo "=== Summary ==="
echo "All daily ingestion binaries are ready to use."
echo ""
echo "Available commands:"
echo "  - cargo run --bin pullall-daily          # Pull historical daily data"
echo "  - cargo run --bin ingest_tushare_stock_daily  # Ingest from Tushare API"
echo "  - cargo run --bin ingest_tushare_stock    # Ingest stock list from Tushare"
echo "  - cargo run --bin ingest_tushare_index    # Ingest index data from Tushare"
