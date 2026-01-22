-- Migration: Create stock_history table
-- Description: Table for storing historical stock data used in inference
-- Date: 2025-12-23

CREATE TABLE IF NOT EXISTS stock_history (
    -- Primary identifiers
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV data
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    
    -- Temporal features
    month INTEGER NOT NULL,
    weekday INTEGER NOT NULL,
    quarter INTEGER,
    
    -- Simple Moving Averages
    sma5 REAL,
    sma20 REAL,
    
    -- Technical Indicators
    rsi REAL,
    
    -- Return & Volume Features
    daily_return REAL,
    volume_ratio REAL,
    
    -- Constraints
    PRIMARY KEY (symbol, timestamp)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_history_symbol ON stock_history(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_history_timestamp ON stock_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_stock_history_symbol_timestamp ON stock_history(symbol, timestamp);

-- Add comments
COMMENT ON TABLE stock_history IS 'Historical stock data for inference and prediction';
COMMENT ON COLUMN stock_history.symbol IS 'Stock ticker symbol';
COMMENT ON COLUMN stock_history.timestamp IS 'Timestamp of the trading period';
