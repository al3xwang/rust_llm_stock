#!/bin/bash

# Update Stock Basic and Trade Calendar from Tushare
# This script ingests stock basic information from Tushare API

set -e  # Exit on error

echo "=== Stock Data Update Script ==="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "Please create a .env file with:"
    echo "  DATABASE_URL=postgresql://user:password@host:port/database"
    echo "  TUSHARE_TOKEN=your_tushare_api_token"
    echo ""
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check required variables
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set in .env file"
    exit 1
fi

if [ -z "$TUSHARE_TOKEN" ] || [ "$TUSHARE_TOKEN" = "your_token_here" ]; then
    echo "❌ TUSHARE_TOKEN not set or using placeholder value"
    echo ""
    echo "Please update .env file with your actual Tushare API token"
    echo "Get your token from: https://tushare.pro/user/token"
    echo ""
    exit 1
fi

echo "✓ Environment variables loaded"
echo "  DATABASE: ${DATABASE_URL%%\?*}"  # Print DB URL without password
echo "  TUSHARE: Token configured"
echo ""

# Check if database is accessible
echo "Checking database connection..."
if psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo "✓ Database connection successful"
else
    echo "⚠️  Could not connect to database"
    echo "   Make sure PostgreSQL is running and credentials are correct"
    echo "   Continuing anyway..."
fi
echo ""

# Build if needed
BINARY_PATH="../target/release/ingest_tushare_stock"
if [ ! -f "$BINARY_PATH" ]; then
    echo "Building ingest_tushare_stock binary..."
    cargo build --release --bin ingest_tushare_stock
    echo ""
fi

# Run stock basic ingestion
echo "===================="
echo "Step 1: Ingesting Stock Basic Data"
echo "===================="
echo ""

if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ Binary not found at $BINARY_PATH"
    echo "   Build may have failed. Check output above."
    exit 1
fi

"$BINARY_PATH"

echo ""
echo "===================="
echo "✓ Stock Basic Update Complete"
echo "===================="
echo ""
echo "Next steps:"
echo "  1. Check the database for updated stock_basic table"
echo "  2. Run daily data ingestion: cargo run --release --bin pullall-daily"
echo "  3. Create ML training datasets: cargo run --release --bin dataset_creator"
echo ""
