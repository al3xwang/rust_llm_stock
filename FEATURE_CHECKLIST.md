# Feature Engineering Progress Checklist

## Status: Database Schema Complete ✅, Code Implementation Complete ✅

### Phase 1: Infrastructure Setup (COMPLETED ✅)

- [x] Create migration file (003_add_five_predictive_features.sql)
- [x] Apply migration to database (5 columns + index created)
- [x] Update FeatureRow struct with 5 new fields
- [x] Add calculation logic for each feature
- [x] Update INSERT statement and bind parameters
- [x] Compile dataset_creator successfully (6.90s, 0 errors)

### Phase 2: Data Population (NEXT STEPS ⏳)

- [ ] **Run dataset_creator to populate historical data**
  ```bash
  cd /Users/alex/stock-analysis-workspace/rust_llm_stock
  cargo run --release --bin dataset_creator
  ```
  - Expected time: 60-90 minutes
  - Records to process: ~7,691,645
  - Stocks: ~5,389

- [ ] **Validate feature population**
  ```sql
  SELECT 
      COUNT(*) as total,
      COUNT(pe_percentile_52w) as pe_populated,
      COUNT(volume_accel_5d) as vol_populated,
      COUNT(price_vs_52w_high) as price_populated,
      COUNT(consecutive_up_days) as consec_populated
  FROM ml_training_dataset 
  WHERE trade_date = '20260113';
  ```
  - Expected: 4,500+ PE, 5,300+ volume, 5,000+ price, 5,371 consecutive

- [ ] **Run sector momentum second-pass UPDATE**
  ```sql
  WITH sector_stats AS (
      SELECT industry, trade_date,
             AVG(price_momentum_5) as sector_momentum_5d
      FROM ml_training_dataset
      WHERE industry IS NOT NULL
      GROUP BY industry, trade_date
  ),
  market_stats AS (
      SELECT trade_date,
             AVG(price_momentum_5) as market_momentum_5d
      FROM ml_training_dataset
      GROUP BY trade_date
  )
  UPDATE ml_training_dataset m
  SET sector_momentum_vs_market = s.sector_momentum_5d - mk.market_momentum_5d
  FROM sector_stats s
  JOIN market_stats mk ON s.trade_date = mk.trade_date
  WHERE m.industry = s.industry 
    AND m.trade_date = s.trade_date;
  ```
  - Expected time: 5-10 minutes
  - Records updated: ~7,691,645

### Phase 3: Model Training (PENDING ⏹️)

- [ ] **Export updated training data**
  ```bash
  cargo run --release --bin export_training_data
  ```
  - Output: training_data.csv with 110 features

- [ ] **Verify CSV columns**
  ```bash
  head -1 training_data.csv | tr ',' '\n' | wc -l
  ```
  - Expected: 111 (110 features + 1 target)

- [ ] **Retrain model with 110 features**
  ```bash
  cargo run --release --features pytorch --bin train
  ```
  - Expected time: 2-4 hours (80-100 epochs)
  - Output: artifacts/best_model.safetensors (new version)

- [ ] **Monitor training metrics**
  - Validation loss: Target <0.015 (vs baseline ~0.018)
  - Early stopping: Should trigger at ~60-80 epochs
  - Watch for: Faster convergence, lower final loss

### Phase 4: Validation (PENDING ⏹️)

- [ ] **Backtest on Jan 12-14 data**
  ```bash
  cargo run --release --features pytorch --bin batch_predict
  ```

- [ ] **Calculate accuracy improvement**
  - Baseline (105 features): 70-75%
  - Target (110 features): 75-80%
  - Minimum acceptable: +5% absolute improvement

- [ ] **Analyze which features helped most**
  - Use feature importance analysis
  - Document in FEATURE_ENGINEERING_RESULTS.md

### Phase 5: Production Deployment (PENDING ⏹️)

- [ ] Update generate_realtime_predictions.sh if needed
- [ ] Schedule daily dataset_creator runs (cron job)
- [ ] Update documentation with new feature details
- [ ] Create monitoring dashboard for feature quality

### Phase 6: Next Feature Batch (PLANNED 🔮)

- [ ] Implement money flow features (institutional vs retail)
- [ ] Add cross-stock correlation (beta to market)
- [ ] Consider order book depth features (if data available)

## Quick Commands Reference

### Check feature data quality
```sql
-- Sample feature values
SELECT ts_code, trade_date,
       pe_percentile_52w,
       sector_momentum_vs_market,
       volume_accel_5d,
       price_vs_52w_high,
       consecutive_up_days
FROM ml_training_dataset 
WHERE trade_date = '20260113' 
  AND pe_percentile_52w IS NOT NULL
LIMIT 10;
```

### Monitor dataset creation progress
```sql
-- Check how many dates have new features populated
SELECT 
    trade_date,
    COUNT(*) as stocks,
    COUNT(pe_percentile_52w) as has_pe,
    COUNT(consecutive_up_days) as has_consec
FROM ml_training_dataset 
GROUP BY trade_date 
ORDER BY trade_date DESC 
LIMIT 10;
```

### Identify problematic stocks
```sql
-- Find stocks with missing features
SELECT ts_code, trade_date,
       CASE WHEN pe_percentile_52w IS NULL THEN 'Missing PE' END as pe_issue,
       CASE WHEN price_vs_52w_high IS NULL THEN 'Missing 52W' END as price_issue
FROM ml_training_dataset 
WHERE trade_date = '20260113'
  AND (pe_percentile_52w IS NULL OR price_vs_52w_high IS NULL)
LIMIT 20;
```

## Troubleshooting Quick Fixes

### Dataset creator running too slow
```bash
# Process only recent data first (test run)
# Note: May need to add --start-date flag support to dataset_creator.rs
cargo run --release --bin dataset_creator 2>&1 | tee dataset_creator.log
```

### Missing PE data
```bash
# Re-pull daily_basic if needed
cargo run --release --bin pullall-daily-basic
```

### Check compilation status
```bash
cargo build --release --bin dataset_creator 2>&1 | grep -i "error\|warning" | tail -20
```

## Success Criteria Summary

| Metric | Baseline | Target | Validation Method |
|--------|----------|--------|-------------------|
| Feature count | 105 | 110 | `SELECT COUNT(*) FROM information_schema.columns WHERE table_name='ml_training_dataset'` |
| PE percentile populated | N/A | >4,500 stocks | `SELECT COUNT(pe_percentile_52w) FROM ml_training_dataset WHERE trade_date='20260113'` |
| Model accuracy | 70-75% | 75-80% | Backtest on Jan 12-14 data |
| Validation loss | ~0.018 | <0.015 | Training logs |
| Training time | 2-3 hours | 2-4 hours | Expect +20% overhead |

## Timeline Estimate

- Phase 2 (Data Population): **1-2 hours** (includes second-pass update)
- Phase 3 (Model Training): **3-4 hours** (export + train)
- Phase 4 (Validation): **30 minutes** (backtest + analysis)
- **Total**: ~5-7 hours end-to-end

## Last Updated
Date: 2026-01-14  
Status: Ready for Phase 2 execution ✅
