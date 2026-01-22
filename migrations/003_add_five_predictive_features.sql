-- Migration: Add 5 new predictive features to ml_training_dataset
-- Description: PE percentile, sector momentum, volume acceleration, 52W high distance, consecutive up days
-- Date: 2026-01-14

-- Add new feature columns
ALTER TABLE ml_training_dataset 
ADD COLUMN IF NOT EXISTS pe_percentile_52w DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS sector_momentum_vs_market DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS volume_accel_5d DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS price_vs_52w_high DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS consecutive_up_days INTEGER;

-- Add comments
COMMENT ON COLUMN ml_training_dataset.pe_percentile_52w IS 'PE TTM percentile rank over 52-week window (0-1 range)';
COMMENT ON COLUMN ml_training_dataset.sector_momentum_vs_market IS 'Sector 5-day momentum minus market 5-day momentum';
COMMENT ON COLUMN ml_training_dataset.volume_accel_5d IS 'Rate of change in volume_ratio over 5 days';
COMMENT ON COLUMN ml_training_dataset.price_vs_52w_high IS 'Current price distance from 52-week high (negative = below high)';
COMMENT ON COLUMN ml_training_dataset.consecutive_up_days IS 'Number of consecutive up days (positive) or down days (negative)';

-- Create index for faster sector aggregation queries
CREATE INDEX IF NOT EXISTS idx_ml_training_industry_date ON ml_training_dataset(industry, trade_date);
