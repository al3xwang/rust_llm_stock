-- Migration: Create trading_signals table
-- Description: Store buy/sell signals generated from predictions
-- Date: 2026-01-18

CREATE TABLE IF NOT EXISTS trading_signals (
    -- Primary identifiers
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    
    -- Signal information
    signal_type VARCHAR(20) NOT NULL,  -- 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    signal_score DOUBLE PRECISION NOT NULL,
    
    -- Prediction data
    predicted_return_1day DOUBLE PRECISION,
    predicted_return_3day DOUBLE PRECISION,
    confidence_1day DOUBLE PRECISION,
    confidence_3day DOUBLE PRECISION,
    
    -- Technical indicators at signal time
    rsi_14 DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    bb_percent_b DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    price_momentum_5 DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    
    -- Fundamental data
    total_mv DOUBLE PRECISION,
    pe_ttm DOUBLE PRECISION,
    pb DOUBLE PRECISION,
    
    -- Stock info (denormalized for query convenience)
    stock_name VARCHAR(100),
    industry VARCHAR(50),
    
    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50),
    
    -- Validation (filled after trade date)
    actual_return_1day DOUBLE PRECISION,
    actual_return_3day DOUBLE PRECISION,
    signal_correct_1day BOOLEAN,
    signal_correct_3day BOOLEAN,
    
    -- Unique constraint: one signal per stock per date per model
    CONSTRAINT uq_signal_stock_date_model UNIQUE (ts_code, trade_date, model_version)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_signals_trade_date ON trading_signals(trade_date);
CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON trading_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_ts_code ON trading_signals(ts_code);
CREATE INDEX IF NOT EXISTS idx_signals_score ON trading_signals(signal_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON trading_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_industry ON trading_signals(industry);

-- Composite index for date + signal type queries
CREATE INDEX IF NOT EXISTS idx_signals_date_type ON trading_signals(trade_date, signal_type);

-- Comments
COMMENT ON TABLE trading_signals IS 'Trading signals generated from ML predictions';
COMMENT ON COLUMN trading_signals.signal_type IS 'Signal type: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL';
COMMENT ON COLUMN trading_signals.signal_score IS 'Composite score 0-100 based on prediction + technicals';
COMMENT ON COLUMN trading_signals.signal_correct_1day IS 'Whether signal direction matched actual 1-day movement';
