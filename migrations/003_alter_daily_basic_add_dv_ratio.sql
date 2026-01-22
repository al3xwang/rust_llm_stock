-- Migration: Add dv_ratio column to daily_basic table
-- Date: 2024-05-30

ALTER TABLE daily_basic
ADD COLUMN IF NOT EXISTS dv_ratio DOUBLE PRECISION;
