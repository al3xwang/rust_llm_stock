# Stock Prediction Validation Guide

## Overview

The `validate_predictions` binary automatically validates stock predictions by comparing predicted returns and directions with actual market outcomes.

## Features

- ✅ **1-Day Validation**: Validates predictions against next trading day's actual data
- ✅ **3-Day Validation**: Validates predictions against data 3 trading days later
- ✅ **Automatic Updates**: Populates `actual_*` columns in `stock_predictions` table
- ✅ **Accuracy Metrics**: Calculates prediction correctness, return errors, and confidence analysis
- ✅ **High Confidence Tracking**: Separate metrics for high-confidence predictions (>0.7)

## Usage

### Build the Binary

```bash
cd rust_llm_stock
cargo build --release --bin validate_predictions
```

### Validate All Pending Predictions

```bash
./target/release/validate_predictions
```

This will:
1. Find all predictions with NULL actual values
2. Check if next-day (or 3-day) data is available
3. Calculate actual returns and directions
4. Update the database with validation results
5. Display accuracy statistics

### Validate Specific Date

```bash
./target/release/validate_predictions --date 20260115
```

### Validate Only 1-Day Predictions

```bash
./target/release/validate_predictions --only-1day
```

### Validate Only 3-Day Predictions

```bash
./target/release/validate_predictions --only-3day
```

### Verbose Output with Detailed Stats

```bash
./target/release/validate_predictions --verbose
```

### Test with Limited Data

```bash
./target/release/validate_predictions --limit 100
```

## Daily Workflow Integration

### Recommended Schedule

**After Market Close (4:00 PM CT + 30 minutes):**

```bash
# 1. Wait for daily data ingestion to complete
# 2. Run validation
cd /home/alex/rust_llm_stock
./target/release/validate_predictions --verbose 2>&1 | tee -a logs/validation_$(date +%Y%m%d).log
```

### Cron Job Example

```cron
# Validate predictions daily at 4:30 PM CT (after market close)
30 16 * * 1-5 cd /home/alex/rust_llm_stock && ./target/release/validate_predictions --verbose >> logs/validation.log 2>&1
```

## Output Example

```
╔══════════════════════════════════════════════════════════╗
║         Stock Predictions Validation Tool               ║
╚══════════════════════════════════════════════════════════╝

=== Validating 1-Day Predictions ===

Found 485 predictions to validate
Validated: 485 predictions    

┌─────────────────────────────────────────────────────┐
│  1-Day Prediction Results                           │
└─────────────────────────────────────────────────────┘

  Total Validated:        485
  Correct:                278 (57.32%)
  Incorrect:              207

  Avg Predicted Return:   0.1234%
  Avg Actual Return:      0.0987%
  Avg Prediction Error:   1.2345%

  High Confidence (>0.7): 89 / 142 (62.68%)

=== Validating 3-Day Predictions ===

Found 342 3-day predictions to validate
Validated: 342 predictions    

┌─────────────────────────────────────────────────────┐
│  3-Day Prediction Results                           │
└─────────────────────────────────────────────────────┘

  Total Validated:        342
  Correct:                195 (57.02%)
  Incorrect:              147

  Avg Predicted Return:   0.3456%
  Avg Actual Return:      0.2987%
  Avg Prediction Error:   2.1234%
```

## Database Updates

### 1-Day Predictions

The tool updates these columns in `stock_predictions`:
- `actual_return` - Actual 1-day return (%)
- `actual_direction` - Actual price direction (UP/DOWN)
- `prediction_correct` - Whether predicted direction matched actual

**Calculation:**
```sql
actual_return = (next_close - current_close) / current_close * 100.0
actual_direction = (next_close > current_close)
prediction_correct = (actual_direction == predicted_direction)
```

### 3-Day Predictions

The tool updates these columns:
- `actual_3day_return` - Actual 3-day return (%)
- `actual_3day_direction` - Actual price direction 3 days later
- `prediction_3day_correct` - Whether predicted direction matched actual

**Calculation:**
```sql
-- Finds 3rd trading day after prediction
actual_3day_return = (day3_close - current_close) / current_close * 100.0
actual_3day_direction = (day3_close > current_close)
prediction_3day_correct = (actual_3day_direction == predicted_3day_direction)
```

## Querying Validation Results

### Overall Accuracy

```sql
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END) as correct_1day,
    AVG(CASE WHEN prediction_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_1day,
    SUM(CASE WHEN prediction_3day_correct THEN 1 ELSE 0 END) as correct_3day,
    AVG(CASE WHEN prediction_3day_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_3day
FROM stock_predictions
WHERE actual_return IS NOT NULL;
```

### Accuracy by Confidence Level

```sql
SELECT 
    CASE 
        WHEN confidence >= 0.8 THEN 'Very High (0.8+)'
        WHEN confidence >= 0.7 THEN 'High (0.7-0.8)'
        WHEN confidence >= 0.6 THEN 'Medium (0.6-0.7)'
        ELSE 'Low (<0.6)'
    END as confidence_level,
    COUNT(*) as total,
    AVG(CASE WHEN prediction_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy
FROM stock_predictions
WHERE actual_return IS NOT NULL
GROUP BY 
    CASE 
        WHEN confidence >= 0.8 THEN 'Very High (0.8+)'
        WHEN confidence >= 0.7 THEN 'High (0.7-0.8)'
        WHEN confidence >= 0.6 THEN 'Medium (0.6-0.7)'
        ELSE 'Low (<0.6)'
    END
ORDER BY MIN(confidence) DESC;
```

### Recent Performance

```sql
SELECT 
    trade_date,
    COUNT(*) as predictions,
    AVG(CASE WHEN prediction_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_1day,
    AVG(CASE WHEN prediction_3day_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_3day,
    AVG(ABS(predicted_return - actual_return)) as avg_error
FROM stock_predictions
WHERE actual_return IS NOT NULL
  AND trade_date >= TO_CHAR(CURRENT_DATE - INTERVAL '30 days', 'YYYYMMDD')
GROUP BY trade_date
ORDER BY trade_date DESC;
```

## Troubleshooting

### No Predictions Found

**Issue**: "Found 0 predictions to validate"

**Possible Causes:**
1. All predictions already validated (`actual_return IS NOT NULL`)
2. No new market data available yet
3. Predictions were generated for future dates

**Solution:**
```bash
# Check if there are unvalidated predictions
psql -c "SELECT COUNT(*) FROM stock_predictions WHERE actual_return IS NULL;"

# Check latest data in adjusted_stock_daily
psql -c "SELECT MAX(trade_date) FROM adjusted_stock_daily;"
```

### Database Connection Issues

**Issue**: "Error connecting to database"

**Solution:**
```bash
# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1;"
```

### Validation Lag (3-Day Predictions)

**Issue**: "3-day predictions not validating"

**Cause**: Not enough trading days have passed yet

**Explanation**: 3-day predictions require 3 **trading days** after the prediction date. Weekends and holidays don't count.

Example:
- Prediction made: Monday Jan 13
- Needs data from: Thursday Jan 16 (3 trading days: Tue, Wed, Thu)
- Available for validation: After Thursday market close

## Performance Notes

- Validation processes ~500-1000 predictions per second
- Uses database indexes on `(ts_code, trade_date)` for fast lookups
- Batches database updates for efficiency
- Progress indicator shows every 100 predictions

## Integration with Daily Pipeline

The complete daily workflow should be:

```bash
#!/bin/bash
# daily_stock_workflow.sh

set -e

echo "=== Daily Stock Analysis Workflow ==="
echo "Started: $(date)"

# 1. Fetch fresh market data (after market close)
echo "Step 1: Fetching daily data..."
cargo run --release --bin pullall-daily

# 2. Update adjusted prices
echo "Step 2: Updating adjusted prices..."
cargo run --release --bin create_adjusted_daily

# 3. Calculate features for new data
echo "Step 3: Calculating features..."
cargo run --release --bin dataset_creator

# 4. Generate predictions for tomorrow (before market open)
echo "Step 4: Generating predictions..."
cargo run --release --bin batch_predict --features pytorch

# 5. Validate yesterday's predictions (after today's data available)
echo "Step 5: Validating predictions..."
cargo run --release --bin validate_predictions --verbose

echo "Completed: $(date)"
```

## Next Steps

After validation is working:
1. Monitor accuracy trends over time
2. Adjust model confidence thresholds
3. Filter predictions by minimum confidence
4. Compare 1-day vs 3-day prediction performance
5. Identify stocks/sectors with best prediction accuracy
