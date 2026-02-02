# Training Data Export Summary

**Export Date:** January 15, 2026  
**Total Dataset Records:** 13,022,655 rows (5,356 unique stocks)  
**Date Range:** 2011-01-04 to 2026-01-14 (15 years)

## Exported Files

| File | Rows | Size | Purpose |
|------|------|------|---------|
| `training_data.csv` | 1,972,361 | 2.0 GB | Complete dataset (1000 random stocks, 7+ years history) |
| `train.csv` | 1,042,869 | 1.1 GB | Training split (~53%) |
| `val.csv` | 298,067 | 320 MB | Validation split (~15%) |
| `test.csv` | 149,318 | 161 MB | Test split (~8%) |

## Data Split Strategy

The export process:
1. **Stock Selection**: Selected 1,000 random stocks with at least 7 years of listing history
2. **Data Quality**: Filtered for stocks with complete `pe`, `pe_ttm`, and `dv_ttm` data
3. **Warmup Period**: First 2 years (480 trading days) used as warmup only (not in train/val/test)
4. **Training Window**: Remaining 5 years split as:
   - **70% Training** → `train.csv`
   - **20% Validation** → `val.csv`  
   - **10% Test** → `test.csv`

## Features (122 columns total)

### Identifiers (2)
- `ts_code`, `trade_date`

### Basic Features (6)
- `industry`, `act_ent_type`, `volume`, `amount`, `month`, `weekday`, `quarter`, `week_no`

### Price Features (11)
- Percentage-based OHLC: `open_pct`, `high_pct`, `low_pct`, `close_pct`
- Intraday movements: `high_from_open_pct`, `low_from_open_pct`, `close_from_open_pct`, `intraday_range_pct`, `close_position_in_range`

### Moving Averages (8)
- EMAs: `ema_5`, `ema_10`, `ema_20`, `ema_30`, `ema_60`
- SMAs: `sma_5`, `sma_10`, `sma_20`

### MACD Indicators (7)
- Daily: `macd_line`, `macd_signal`, `macd_histogram`
- Weekly: `macd_weekly_line`, `macd_weekly_signal`
- Monthly: `macd_monthly_line`, `macd_monthly_signal`

### Technical Indicators (19)
- RSI: `rsi_14`
- KDJ: `kdj_k`, `kdj_d`, `kdj_j`
- Bollinger Bands: `bb_upper`, `bb_middle`, `bb_lower`, `bb_bandwidth`, `bb_percent_b`
- Volatility: `atr`, `volatility_5`, `volatility_20`, `asi`, `obv`, `volume_ratio`
- Momentum: `price_momentum_5`, `price_momentum_10`, `price_momentum_20`, `price_position_52w`

### Candlestick Features (7)
- Sizes: `body_size`, `upper_shadow`, `lower_shadow`
- Patterns: `is_doji`, `is_hammer`, `is_shooting_star`
- Trend: `consecutive_days`

### Advanced Indicators (8)
- `trend_strength`, `adx_14`, `vwap_distance_pct`, `cmf_20`
- `williams_r_14`, `aroon_up_25`, `aroon_down_25`

### Lagged Features (9)
- Returns: `return_lag_1`, `return_lag_2`, `return_lag_3`
- Gaps: `overnight_gap`, `gap_pct`
- Volume: `volume_roc_5`, `volume_spike`
- Price: `price_roc_5`, `price_roc_10`, `price_roc_20`, `hist_volatility_20`

### Market Index Features (18)
- **CSI300**: `index_csi300_pct_chg`, `index_csi300_vs_ma5_pct`, `index_csi300_vs_ma20_pct`
- **ChiNext**: `index_chinext_pct_chg`, `index_chinext_vs_ma5_pct`, `index_chinext_vs_ma20_pct`
- **XIN9**: `index_xin9_pct_chg`, `index_xin9_vs_ma5_pct`, `index_xin9_vs_ma20_pct`
- **HSI**: `index_hsi_pct_chg`, `index_hsi_vs_ma5_pct`, `index_hsi_vs_ma20_pct`
- **USDCNH FX**: `fx_usdcnh_pct_chg`, `fx_usdcnh_vs_ma5_pct`, `fx_usdcnh_vs_ma20_pct`

### Money Flow Features (4)
- `net_mf_vol`, `net_mf_amount`, `smart_money_ratio`, `large_order_flow`

### Fundamental Features (13)
- Valuation: `pe`, `pe_ttm`, `pb`, `ps`, `ps_ttm`
- Dividend: `dv_ratio`, `dv_ttm`
- Shares: `total_share`, `float_share`, `free_share`
- Market Cap: `total_mv`, `circ_mv`
- Turnover: `turnover_rate`, `turnover_rate_f`

### Volatility Regime (2)
- `vol_percentile`, `high_vol_regime`

### Advanced Predictive Features (5)
- `pe_percentile_52w` - PE ratio position in 52-week range
- `sector_momentum_vs_market` - Sector performance vs market
- `volume_accel_5d` - Volume acceleration trend
- `price_vs_52w_high` - Distance from 52-week high
- `consecutive_up_days` - Consecutive up/down days count

### Target Variables (4)
- `next_day_return` - Next trading day return (regression target)
- `next_day_direction` - Next day direction (classification target)
- `next_3day_return` - 3-day forward return
- `next_3day_direction` - 3-day direction

## Training on Server

### File Upload
Upload these files to your training server:
```bash
scp train.csv val.csv test.csv user@server:/path/to/training/data/
```

### PyTorch Training Command
From the project directory on server:
```bash
cargo run --release --features pytorch --bin train -- \
  --train data/train.csv \
  --val data/val.csv \
  --device cuda
```

### Model Configuration
- **Input features**: 118 features (excludes identifiers and targets)
- **Sequence length**: 30 days
- **Target**: `next_day_return` (percentage return)
- **Architecture**: LSTM + Transformer hybrid
- **Batch size**: 48
- **Learning rate**: 1e-4 with decay

## Data Quality Notes

✅ **Strengths:**
- 15 years of historical data (2011-2026)
- 122 comprehensive features including fundamentals, technicals, and market indices
- Quality filtering: Only stocks with complete fundamental data
- Proper train/val/test split with 2-year warmup period
- Percentage-based price features (scale-invariant)

⚠️ **Considerations:**
- 35 stocks skipped due to insufficient data (< 60 days)
- Export limited to 1,000 stocks with 7+ years history (quality over quantity)
- Missing data handled: Empty strings in CSV (model should handle NaN gracefully)

## Next Steps

1. **Upload files** to your training server
2. **Verify PyTorch environment** has libtorch installed
3. **Run training** with `--features pytorch` flag
4. **Monitor** via TensorBoard logs
5. **Evaluate** on test.csv after training completes

---

**Files Location:** `/Users/alex/stock-analysis-workspace/rust_llm_stock/`
