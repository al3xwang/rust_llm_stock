-- Migration: Add missing index columns to ml_training_dataset
-- Description: Add XIN9 (FTSE China A50), HSI (Hang Seng Index), and USDCNH FX columns
-- Date: 2025-12-28

ALTER TABLE ml_training_dataset
    ADD COLUMN IF NOT EXISTS index_xin9_pct_chg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS index_xin9_vs_ma5_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS index_xin9_vs_ma20_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS index_hsi_pct_chg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS index_hsi_vs_ma5_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS index_hsi_vs_ma20_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fx_usdcnh_pct_chg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fx_usdcnh_vs_ma5_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS fx_usdcnh_vs_ma20_pct DOUBLE PRECISION;

-- Add comments
COMMENT ON COLUMN ml_training_dataset.index_xin9_pct_chg IS 'FTSE China A50 Index daily percent change';
COMMENT ON COLUMN ml_training_dataset.index_xin9_vs_ma5_pct IS 'FTSE China A50 Index vs 5-day MA percent';
COMMENT ON COLUMN ml_training_dataset.index_xin9_vs_ma20_pct IS 'FTSE China A50 Index vs 20-day MA percent';
COMMENT ON COLUMN ml_training_dataset.index_hsi_pct_chg IS 'Hang Seng Index daily percent change';
COMMENT ON COLUMN ml_training_dataset.index_hsi_vs_ma5_pct IS 'Hang Seng Index vs 5-day MA percent';
COMMENT ON COLUMN ml_training_dataset.index_hsi_vs_ma20_pct IS 'Hang Seng Index vs 20-day MA percent';
COMMENT ON COLUMN ml_training_dataset.fx_usdcnh_pct_chg IS 'USD/CNH FX rate daily percent change';
COMMENT ON COLUMN ml_training_dataset.fx_usdcnh_vs_ma5_pct IS 'USD/CNH FX rate vs 5-day MA percent';
COMMENT ON COLUMN ml_training_dataset.fx_usdcnh_vs_ma20_pct IS 'USD/CNH FX rate vs 20-day MA percent';
