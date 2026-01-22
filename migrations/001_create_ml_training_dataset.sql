-- Migration: Create ml_training_dataset table
-- Description: Main table for stock training data with OHLCV and technical indicators
-- Date: 2025-12-23

CREATE TABLE IF NOT EXISTS ml_training_dataset (
    -- Primary identifiers
    ts_code VARCHAR(20) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    
    -- OHLCV data
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    amount DOUBLE PRECISION,
    
    -- Temporal features
    weekday SMALLINT,
    week_no SMALLINT,
    quarter SMALLINT,
    
    -- Exponential Moving Averages
    ema_5 DOUBLE PRECISION,
    ema_10 DOUBLE PRECISION,
    ema_20 DOUBLE PRECISION,
    ema_30 DOUBLE PRECISION,
    ema_60 DOUBLE PRECISION,
    
    -- Simple Moving Averages
    sma_5 DOUBLE PRECISION,
    sma_10 DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    
    -- MACD Indicators (Daily)
    macd_line DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    
    -- MACD Indicators (Weekly)
    macd_weekly_line DOUBLE PRECISION,
    macd_weekly_signal DOUBLE PRECISION,
    
    -- MACD Indicators (Monthly)
    macd_monthly_line DOUBLE PRECISION,
    macd_monthly_signal DOUBLE PRECISION,
    
    -- Technical Indicators
    rsi_14 DOUBLE PRECISION,
    cci_14 DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    asi DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    
    -- Bollinger Bands
    bb_upper DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_bandwidth DOUBLE PRECISION,
    
    -- Return & Volume Features
    pct_change DOUBLE PRECISION,
    daily_return DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    
    -- Target variable
    next_day_return DOUBLE PRECISION,
    
    -- Constraints
    PRIMARY KEY (ts_code, trade_date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ml_training_trade_date ON ml_training_dataset(trade_date);
CREATE INDEX IF NOT EXISTS idx_ml_training_ts_code ON ml_training_dataset(ts_code);
CREATE INDEX IF NOT EXISTS idx_ml_training_ts_code_trade_date ON ml_training_dataset(ts_code, trade_date);

-- Add comments
COMMENT ON TABLE ml_training_dataset IS 'Stock market training data with OHLCV and technical indicators';
COMMENT ON COLUMN ml_training_dataset.ts_code IS 'Stock ticker symbol (e.g., 000001.SZ)';
COMMENT ON COLUMN ml_training_dataset.trade_date IS 'Trading date in YYYYMMDD format';
COMMENT ON COLUMN ml_training_dataset.next_day_return IS 'Target variable: next trading day return';
