```sqlite
-- Migration: Create daily_basic table with all Tushare daily_basic fields
-- Date: 2024-05-30

CREATE TABLE IF NOT EXISTS daily_basic (
    ts_code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    close DOUBLE PRECISION,
    turnover_rate DOUBLE PRECISION,
    turnover_rate_f DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    pe DOUBLE PRECISION,
    pe_ttm DOUBLE PRECISION,
    pb DOUBLE PRECISION,
    ps DOUBLE PRECISION,
    ps_ttm DOUBLE PRECISION,
    dv_ratio DOUBLE PRECISION,
    dv_ttm DOUBLE PRECISION,
    total_share DOUBLE PRECISION,
    float_share DOUBLE PRECISION,
    free_share DOUBLE PRECISION,
    total_mv DOUBLE PRECISION,
    circ_mv DOUBLE PRECISION,
    PRIMARY KEY (ts_code, trade_date)
);

-- If you are inserting from text (e.g. CSV or JSON), use explicit casts in your INSERT statement:
-- Example: INSERT INTO daily_basic (pe, ...) VALUES ($1::double precision, ...)
```
