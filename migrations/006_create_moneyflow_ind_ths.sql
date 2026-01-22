
-- Create table for industry money flow data from Tushare (moneyflow_ind_ths)

CREATE TABLE IF NOT EXISTS moneyflow_ind_ths (
    ts_code VARCHAR(20) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    industry_name VARCHAR(128),
    net_buy_amount DOUBLE PRECISION,
    net_sell_amount DOUBLE PRECISION,
    net_amount DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_moneyflow_ind_ths_trade_date ON moneyflow_ind_ths(trade_date);
CREATE INDEX IF NOT EXISTS idx_moneyflow_ind_ths_industry_name ON moneyflow_ind_ths(industry_name);

