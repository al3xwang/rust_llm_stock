-- Migration: Create stock_daily_adjusted table
-- Description: Store forward-adjusted (qfq) stock prices accounting for splits and dividends
-- Date: 2025-12-25

CREATE TABLE IF NOT EXISTS stock_daily_adjusted (
    -- Primary identifiers
    ts_code VARCHAR(20) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    
    -- Forward-adjusted OHLCV data (qfq - 前复权)
    -- These prices are adjusted for all stock splits, dividends, and rights offerings
    -- providing continuous, comparable price series across time
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    pre_close DOUBLE PRECISION,
    
    -- Volume and amount (not adjusted)
    volume DOUBLE PRECISION NOT NULL,
    amount DOUBLE PRECISION,
    
    -- Price changes (based on adjusted prices)
    change DOUBLE PRECISION,
    pct_chg DOUBLE PRECISION,
    
    -- Adjustment factor used by Tushare
    adj_factor DOUBLE PRECISION,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    PRIMARY KEY (ts_code, trade_date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_trade_date ON stock_daily_adjusted(trade_date);
CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_ts_code ON stock_daily_adjusted(ts_code);
CREATE INDEX IF NOT EXISTS idx_stock_daily_adjusted_ts_code_date ON stock_daily_adjusted(ts_code, trade_date DESC);

-- Add comments
COMMENT ON TABLE stock_daily_adjusted IS 'Forward-adjusted (qfq) stock prices accounting for splits and dividends';
COMMENT ON COLUMN stock_daily_adjusted.ts_code IS 'Stock ticker symbol (e.g., 000001.SZ)';
COMMENT ON COLUMN stock_daily_adjusted.trade_date IS 'Trading date in YYYYMMDD format';
COMMENT ON COLUMN stock_daily_adjusted.open IS 'Forward-adjusted opening price';
COMMENT ON COLUMN stock_daily_adjusted.close IS 'Forward-adjusted closing price';
COMMENT ON COLUMN stock_daily_adjusted.adj_factor IS 'Adjustment factor from Tushare API';
