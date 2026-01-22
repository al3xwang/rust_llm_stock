#!/bin/bash

# Migrate stock_basic table to include all columns needed for Tushare data

export $(cat .env | xargs)

if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set in .env"
    exit 1
fi

echo "Migrating stock_basic table schema..."
echo ""

# Add missing columns to stock_basic table
psql "$DATABASE_URL" << EOF
-- Add columns if they don't exist
ALTER TABLE stock_basic
ADD COLUMN IF NOT EXISTS area VARCHAR(50),
ADD COLUMN IF NOT EXISTS fullname VARCHAR(200),
ADD COLUMN IF NOT EXISTS enname VARCHAR(200),
ADD COLUMN IF NOT EXISTS cnspell VARCHAR(50),
ADD COLUMN IF NOT EXISTS exchange VARCHAR(20),
ADD COLUMN IF NOT EXISTS curr_type VARCHAR(10),
ADD COLUMN IF NOT EXISTS list_status VARCHAR(1),
ADD COLUMN IF NOT EXISTS delist_date VARCHAR(20),
ADD COLUMN IF NOT EXISTS is_hs VARCHAR(1),
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_stock_basic_symbol ON stock_basic(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_basic_market ON stock_basic(market);
CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
CREATE INDEX IF NOT EXISTS idx_stock_basic_list_status ON stock_basic(list_status);
CREATE INDEX IF NOT EXISTS idx_stock_basic_list_date ON stock_basic(list_date);

-- Show the updated table structure
\d stock_basic

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Table migration completed successfully"
    echo ""
    echo "Next step: Run ./update_stock_data.sh to ingest stock basic data"
else
    echo ""
    echo "❌ Migration failed"
    exit 1
fi
