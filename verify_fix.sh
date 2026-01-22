#!/bin/bash

# Verify the fix for NULL values in technical indicators
# This checks if 20260113 data now has proper technical indicator values

echo "========================================="
echo "Verifying 20260113 Data Quality"
echo "========================================="

psql -U postgres -h 127.0.0.1 -d research <<EOF

-- Check overall feature coverage
SELECT 
  COUNT(*) as total_records,
  COUNT(open_pct) as has_open_pct,
  COUNT(ema_5) as has_ema_5,
  COUNT(rsi_14) as has_rsi_14,
  COUNT(kdj_k) as has_kdj_k,
  COUNT(bb_upper) as has_bb_upper,
  COUNT(atr) as has_atr,
  COUNT(pe_percentile_52w) as has_pe_percentile,
  COUNT(volume_accel_5d) as has_volume_accel,
  COUNT(price_vs_52w_high) as has_price_vs_52w_high,
  COUNT(consecutive_up_days) as has_consecutive_up_days
FROM ml_training_dataset 
WHERE trade_date = '20260113';

-- Sample records with technical indicators
SELECT 
  ts_code,
  trade_date,
  ROUND(open_pct::numeric, 3) as open_pct,
  ROUND(ema_5::numeric, 3) as ema_5,
  ROUND(rsi_14::numeric, 2) as rsi_14,
  ROUND(kdj_k::numeric, 2) as kdj_k,
  ROUND(bb_upper::numeric, 3) as bb_upper,
  ROUND(atr::numeric, 3) as atr,
  ROUND(pe_percentile_52w::numeric, 4) as pe_pct,
  ROUND(volume_accel_5d::numeric, 4) as vol_accel,
  consecutive_up_days
FROM ml_training_dataset 
WHERE trade_date = '20260113'
ORDER BY ts_code
LIMIT 5;

EOF

echo ""
echo "========================================="
echo "Summary:"
echo "If rsi_14, bb_upper, and atr counts match total_records,"
echo "the fix is successful!"
echo "========================================="
