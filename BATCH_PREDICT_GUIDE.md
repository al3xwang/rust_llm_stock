# Running batch_predict - Quick Reference

## Binary Location

```bash
/Users/alex/stock-analysis-workspace/target/release/batch_predict
```

## Basic Usage

### Full Run (All Active Stocks, All Dates)

```bash
/Users/alex/stock-analysis-workspace/target/release/batch_predict
```

### With Custom Concurrency

```bash
# 16 concurrent predictions (optimized for modern hardware)
/Users/alex/stock-analysis-workspace/target/release/batch_predict --concurrency 16

# Maximum concurrency (let system decide)
/Users/alex/stock-analysis-workspace/target/release/batch_predict --concurrency 32

# Conservative (slower, less resource usage)
/Users/alex/stock-analysis-workspace/target/release/batch_predict --concurrency 4
```

### Output Only New Predictions

To avoid redundant predictions and save time, only predict for latest date:

```bash
/Users/alex/stock-analysis-workspace/target/release/batch_predict --output-only-new
```

## What It Does

1. **Retrieves Stock List**
   - Gets all actively traded stocks from `stock_basic` table
   - Filters to only stocks with data in `ml_training_dataset`
   - Found ~5353 active stocks in recent test

2. **Loads Model**
   - Loads best model from `artifacts/best_model.safetensors`
   - Requires PyTorch features enabled (already compiled with `--features pytorch`)

3. **Predicts for Each Stock**
   - Fetches latest 30+ days of features for each stock
   - Runs inference to predict next trading day return
   - Calculates direction (up/down) and confidence

4. **Saves Predictions**
   - Stores in `stock_predictions` table
   - Includes: ts_code, trade_date, predicted_return, predicted_direction, confidence
   - 3-day predictions stored as NULL (model doesn't support yet)

## Environment Variables

Must be set in `.env` file in workspace root:

```bash
DATABASE_URL=postgresql://postgres:12341234@127.0.0.1:5432/research
RUST_LOG=info,rust_llm_stock=debug
```

## Performance Expectations

### Test Run (50 stocks, concurrency=16)

```
Time: ~30-45 seconds
Throughput: ~1-2 stocks/second
Database connections: ~16 active
Memory: ~500 MB - 1 GB
CPU: ~60-70% utilization (depends on CPU cores)
```

### Full Run (5353 stocks)

```
Estimated time: ~40-60 minutes with --concurrency 16
Predictions generated: ~5353 (one per stock for latest date)
Database inserts: ~5353 rows
```

## Monitoring Execution

### In Another Terminal - Watch Database

```bash
# Monitor new predictions being saved in real-time
watch 'psql -h 127.0.0.1 -U postgres -d research \
  -c "SELECT COUNT(*) as predictions, 
             MAX(prediction_date) as latest_prediction 
      FROM stock_predictions 
      WHERE prediction_date > NOW() - INTERVAL 1 hour;"'
```

### Check Progress

```bash
# Count predictions by date
psql -h 127.0.0.1 -U postgres -d research \
  -c "SELECT COUNT(*) as count, MAX(trade_date) as latest 
      FROM stock_predictions 
      GROUP BY trade_date 
      ORDER BY trade_date DESC LIMIT 5;"
```

## Common Issues & Solutions

### Issue: "Connection pool exhausted"

**Cause**: Too many concurrent connections requested
**Solution**: Reduce concurrency
```bash
/Users/alex/stock-analysis-workspace/target/release/batch_predict --concurrency 8
```

### Issue: "No predictions generated"

**Cause**: Stock data may not exist for latest date
**Solution**: Check if market was open
```bash
psql -h 127.0.0.1 -U postgres -d research \
  -c "SELECT MAX(trade_date) FROM ml_training_dataset;"
```

### Issue: "Model not found: artifacts/best_model.safetensors"

**Cause**: Model file missing
**Solution**: Train model first
```bash
cd /Users/alex/stock-analysis-workspace
cargo run --release --features pytorch --bin train
```

### Issue: "Too slow" or "High memory usage"

**Solution**: 
1. Reduce concurrency: `--concurrency 8`
2. Run on smaller subset (if you modify code)
3. Increase system resources if possible

## Database Schema Check

Verify the database has all required tables:

```bash
# Check stock_predictions table exists
psql -h 127.0.0.1 -U postgres -d research -c "\d stock_predictions"

# Check ml_training_dataset has recent data
psql -h 127.0.0.1 -U postgres -d research \
  -c "SELECT COUNT(*) as total_records, 
             COUNT(DISTINCT ts_code) as unique_stocks,
             MAX(trade_date) as latest_date
      FROM ml_training_dataset;"

# Check for model
ls -lh /Users/alex/stock-analysis-workspace/rust_llm_stock/artifacts/best_model.safetensors
```

## Advanced: Understanding Output

When batch_predict runs, you'll see output like:

```
Processing 5353 active stocks...
Starting batch prediction with 16 concurrent workers...
[1/5353] 000001.SZ: 50 records, 1-day pred: 0.0234 (dir: ↑, conf: 0.789)
[2/5353] 000002.SZ: 50 records, 1-day pred: -0.0156 (dir: ↓, conf: 0.612)
...
```

**Interpretation:**
- `50 records`: Used last 50 trading days of features
- `1-day pred: 0.0234`: Predicted next-day return of +2.34%
- `dir: ↑`: Predicted price will go up
- `conf: 0.789`: Confidence score of 78.9%

## Database Predictions Query

After batch_predict completes, query results:

```sql
-- Latest predictions (top 10 by confidence)
SELECT ts_code, trade_date, predicted_return, 
       CASE WHEN predicted_direction THEN '↑' ELSE '↓' END as direction,
       ROUND(confidence::numeric, 3) as confidence
FROM stock_predictions
WHERE trade_date = (SELECT MAX(trade_date) FROM stock_predictions)
ORDER BY confidence DESC
LIMIT 10;

-- Predictions for specific stock
SELECT * FROM stock_predictions 
WHERE ts_code = '000001.SZ' 
ORDER BY trade_date DESC 
LIMIT 5;

-- Accuracy metrics (when actual returns available)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct,
    ROUND(100.0 * SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) / 
        COUNT(*)::numeric, 2) as accuracy_percent
FROM stock_predictions
WHERE actual_return IS NOT NULL
  AND prediction_date > NOW() - INTERVAL '30 days';
```

## Deployment Checklist

- [ ] Binary compiled: `/Users/alex/stock-analysis-workspace/target/release/batch_predict`
- [ ] Database connection working: `DATABASE_URL` set and tested
- [ ] Model file exists: `artifacts/best_model.safetensors`
- [ ] Latest features in database: `ml_training_dataset` has recent data
- [ ] stock_predictions table empty or ready for new data
- [ ] Concurrency setting chosen (recommend 16 for modern hardware)

---

**Last Updated**: 2025-01-16
**Status**: Ready to Deploy
