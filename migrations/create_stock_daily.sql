-- Create stock_daily table compatible with Tushare Pro 'daily' API
CREATE TABLE IF NOT EXISTS stock_daily (
    ts_code      VARCHAR(16)  NOT NULL, -- Stock code
    trade_date   VARCHAR(8)   NOT NULL, -- Trading date (YYYYMMDD)
    open         DOUBLE PRECISION,      -- Open price
    high         DOUBLE PRECISION,      -- High price
    low          DOUBLE PRECISION,      -- Low price
    close        DOUBLE PRECISION,      -- Close price
    pre_close    DOUBLE PRECISION,      -- Previous close price
    change       DOUBLE PRECISION,      -- Price change
    pct_chg      DOUBLE PRECISION,      -- Percentage change
    vol          DOUBLE PRECISION,      -- Volume (in 100 shares)
    amount       DOUBLE PRECISION,      -- Amount (in thousand RMB)
    PRIMARY KEY (ts_code, trade_date)
);

-- Index for fast queries by date
CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date ON stock_daily(trade_date);
-- Index for fast queries by stock
CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code ON stock_daily(ts_code);
