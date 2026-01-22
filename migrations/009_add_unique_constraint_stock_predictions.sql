-- Migration: Enforce unique predictions per stock/date/model_version
-- Description: Deduplicate existing rows and add a unique index on (ts_code, trade_date, model_version)
-- Date: 2026-01-17

-- Normalize model_version: set default '' and replace NULLs
ALTER TABLE stock_predictions
    ALTER COLUMN model_version SET DEFAULT '';

UPDATE stock_predictions
SET model_version = ''
WHERE model_version IS NULL;

-- Deduplicate: keep the latest prediction per (ts_code, trade_date, model_version)
WITH ranked AS (
    SELECT ctid,
           ts_code,
           trade_date,
           model_version,
           prediction_date,
           ROW_NUMBER() OVER (
               PARTITION BY ts_code, trade_date, model_version
               ORDER BY prediction_date DESC
           ) AS rn
    FROM stock_predictions
)
DELETE FROM stock_predictions sp
USING ranked r
WHERE sp.ctid = r.ctid
  AND r.rn > 1;

-- Create unique index (idempotent)
CREATE UNIQUE INDEX IF NOT EXISTS ux_stock_predictions_unique
    ON stock_predictions (ts_code, trade_date, model_version);

COMMENT ON INDEX ux_stock_predictions_unique IS 'Ensures one prediction per (ts_code, trade_date, model_version)';
-- Migration: Add unique key for stock_predictions upserts
-- Description: Enforce one prediction per (ts_code, trade_date, model_version) and deduplicate history
-- Date: 2025-01-05

BEGIN;

-- Normalize model_version to allow deterministic unique index
UPDATE stock_predictions
SET model_version = ''
WHERE model_version IS NULL;

ALTER TABLE stock_predictions
    ALTER COLUMN model_version SET DEFAULT '',
    ALTER COLUMN model_version SET NOT NULL;

-- Deduplicate existing rows, keeping the most recent prediction per key
WITH ranked AS (
    SELECT ctid,
           ts_code,
           trade_date,
           model_version,
           prediction_date,
           ROW_NUMBER() OVER (
               PARTITION BY ts_code, trade_date, model_version
               ORDER BY prediction_date DESC
           ) AS rn
    FROM stock_predictions
)
DELETE FROM stock_predictions sp
USING ranked r
WHERE sp.ctid = r.ctid
  AND r.rn > 1;

-- Enforce uniqueness for upserts
CREATE UNIQUE INDEX IF NOT EXISTS ux_stock_predictions_unique
ON stock_predictions (ts_code, trade_date, model_version);

COMMIT;
