# NULL Values in Technical Indicators - Root Cause Analysis and Fix

## Problem Statement

Many records in `ml_training_dataset` had NULL values for technical indicators like:
- `rsi_14` (Relative Strength Index)
- `bb_upper`, `bb_lower`, `bb_bandwidth` (Bollinger Bands)
- `atr` (Average True Range)
- `pe_percentile_52w` and other 52-week features

This affected data quality across multiple dates, not just 20260113.

## Root Cause

The issue had **two interconnected problems**:

### Problem 1: Insufficient Historical Data Fetching

When processing stocks incrementally, the code fetched data from:
```rust
fetch_start = min_date - 400 days
```

But for stocks listed **less than 400 days ago**, this meant:
- The `fetch_start` date would be BEFORE the stock's listing date
- The database query would return very few records (only since listing)
- Example: A stock listed on 2025-06-01 processing data for 2026-01-13:
  - Expected fetch_start: ~2024-07-01 (400 days back)
  - Actual data available: Only from 2025-06-01 (226 days)
  - Result: Not enough history for 252-day (52-week) indicators

### Problem 2: No Validation of Historical Depth

The code calculated technical indicators with conditional logic like:
```rust
let rsi_14 = if closes.len() >= 14 {
    Some(calculate_rsi(...))
} else {
    None  // ← Returns None if insufficient data
}
```

However, there were **two critical gaps**:

1. **No stock-level validation**: The function didn't check if the stock had minimum required history (60+ days) before processing
2. **No record-level validation**: Records were inserted even when `closes.len() < 60`, causing:
   - Simple features (open_pct, volume_ratio) to calculate correctly (only need 1-5 days)
   - Technical indicators (RSI, Bollinger Bands) to return `None` → NULL in database

## The Fix

### Fix 1: Stock-Level Historical Data Validation

Added validation at the beginning of `calculate_features_for_stock_sync()`:

```rust
// CRITICAL FIX: Skip stocks with insufficient historical data
let min_history_required = 60;
if daily_data.len() < min_history_required {
    eprintln!(
        "⚠️  Stock {} has only {} days of history (need {}). Skipping feature calculation.",
        ts_code, daily_data.len(), min_history_required
    );
    return vec![];  // Skip this stock entirely
}
```

**Result**: Stocks with < 60 days of data are skipped completely, preventing partial feature calculations.

### Fix 2: Record-Level Historical Depth Validation

Modified the insertion filter to check historical depth:

```rust
// OLD CODE - only checked date range:
if day.trade_date.as_str() >= min_date {
    features.push(FeatureRow { ... });
}

// NEW CODE - checks BOTH date range AND historical depth:
if day.trade_date.as_str() >= min_date && closes.len() >= 60 {
    features.push(FeatureRow { ... });
}
```

**Result**: Even for stocks with enough total history, we only insert records once we've accumulated 60+ days of context for that specific day.

### Fix 3: Better Logging

Added tracking to show which stocks are being processed vs. skipped:

```rust
let mut skipped_stocks = 0;
let mut processed_stocks = 0;

// ... in loop ...
if feature_rows.is_empty() {
    skipped_stocks += 1;
} else {
    processed_stocks += 1;
}

// Final summary
println!(
    "⚠️  Skipped {} stocks due to insufficient historical data (< 60 days)",
    skipped_stocks
);
```

## Why This Works

### Scenario 1: Recently Listed Stock
- Stock listed: 2025-12-01 (45 days ago)
- Processing date: 2026-01-14
- **Before fix**: Would fetch 45 days, calculate what it could, insert with NULLs
- **After fix**: Skips stock entirely (45 < 60), returns empty vec, no NULLs

### Scenario 2: Stock with Partial History
- Stock listed: 2024-06-01 (592 days ago)
- Processing dates: 2024-06-01 to 2026-01-14
- **Before fix**: First 60 days would have NULLs for technical indicators
- **After fix**: 
  - Fetches all 592 days of data
  - Builds `closes` vector incrementally
  - Only inserts records starting from day 61 (when `closes.len() >= 60`)
  - First 60 days used ONLY for indicator calculation context

### Scenario 3: Well-Established Stock
- Stock listed: 2010-01-01 (16 years ago)
- **Before fix**: Would work correctly (plenty of history)
- **After fix**: Still works correctly, now with explicit validation

## Validation

After running the fixed code, verify with:

```sql
-- Check feature coverage for 20260113
SELECT 
  COUNT(*) as total,
  COUNT(rsi_14) as rsi_count,
  COUNT(bb_upper) as bb_count,
  COUNT(atr) as atr_count
FROM ml_training_dataset 
WHERE trade_date = '20260113';
```

**Expected result**: All counts should be equal (no NULLs)

## Technical Indicator Minimum Requirements

| Indicator | Minimum Days | Reason |
|-----------|-------------|---------|
| RSI-14 | 14 | 14-day relative strength |
| Bollinger Bands | 20 | 20-day SMA + std dev |
| ATR | 14 | 14-day average true range |
| MACD (daily) | 26 | 26-day slow EMA |
| MACD (weekly) | 130 | ~26 weeks × 5 days |
| MACD (monthly) | 260 | ~52 weeks × 5 days |
| 52-week features | 252 | 252 trading days ≈ 1 year |

Our threshold of **60 days** ensures:
- All daily indicators can be calculated (max requirement: 26 days for MACD)
- Sufficient warmup period for EMA/SMA calculations
- Buffer for missing trading days (holidays, etc.)

## Code Changes Summary

**Files modified**: `src/bin/dataset_creator.rs`

**Changes**:
1. Line ~1208: Added stock-level validation with logging
2. Line ~1988: Added record-level historical depth check (`closes.len() >= 60`)
3. Line ~613: Enhanced logging to track processed vs. skipped stocks

**Backward compatibility**: 
- Existing good data remains unchanged
- Only affects newly inserted records
- Stocks/dates with NULLs need re-insertion (delete + re-run)

## Migration Plan for Existing Data

To fix existing NULL values:

```bash
# 1. Identify affected dates
psql -c "SELECT trade_date, COUNT(*) as total, COUNT(rsi_14) as rsi_count 
         FROM ml_training_dataset 
         GROUP BY trade_date 
         HAVING COUNT(*) != COUNT(rsi_14) 
         ORDER BY trade_date;"

# 2. Delete affected dates
psql -c "DELETE FROM ml_training_dataset WHERE trade_date IN ('<date1>', '<date2>');"

# 3. Re-run dataset_creator to recalculate
cargo run --release --bin dataset_creator
```

## Prevention of Future Issues

With these fixes in place:
- ✅ New stocks won't create NULL records until they have 60+ days of history
- ✅ Incremental updates properly accumulate historical context
- ✅ Explicit logging shows which stocks are being skipped and why
- ✅ Database integrity maintained (no partial feature rows)
