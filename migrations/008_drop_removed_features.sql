-- Migration: Drop removed features from ml_training_dataset
-- This migration drops columns that were removed from the dataset_creator and are no longer produced.
-- Back up data before running in production.

BEGIN;

ALTER TABLE IF EXISTS ml_training_dataset
    DROP COLUMN IF EXISTS ps,
    DROP COLUMN IF EXISTS ps_ttm,
    DROP COLUMN IF EXISTS total_mv,
    DROP COLUMN IF EXISTS circ_mv,
    DROP COLUMN IF EXISTS williams_r_14,
    DROP COLUMN IF EXISTS aroon_down_25,
    DROP COLUMN IF EXISTS ema_5,
    DROP COLUMN IF EXISTS ema_10,
    DROP COLUMN IF EXISTS ema_20,
    DROP COLUMN IF EXISTS ema_30,
    DROP COLUMN IF EXISTS ema_60;

COMMIT;