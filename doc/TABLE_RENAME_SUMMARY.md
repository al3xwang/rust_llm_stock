# Daily Table Rename - stock_daily

## Summary of Changes

Successfully renamed the `daily` table to `stock_daily` throughout the codebase.

### Files Modified

1. **src/bin/pullall-daily.rs** - 3 changes:
   - Updated MIN(trade_date) SELECT query
   - Updated MAX(trade_date) SELECT query  
   - Updated INSERT statement

2. **src/bin/create_adjusted_stock_daily.rs** - 1 change:
   - Updated FROM clause in data fetch query

3. **src/bin/create_testing_dataset.rs** - 1 change:
   - Updated FROM clause in dataset query

4. **INGESTION_GUIDE.md** - 2 changes:
   - Updated SQL examples in documentation
   - Changed query references from `daily` to `stock_daily`

### Database Migration

To rename the table in your PostgreSQL database, run:

```bash
chmod +x rename_daily_table.sh
./rename_daily_table.sh
```

This script will:
- Rename the `daily` table to `stock_daily`
- Rename all associated indexes
- Verify the rename was successful

### Verification

All affected binaries have been verified to compile successfully:
- ✓ pullall-daily
- ✓ create_adjusted_stock_daily
- ✓ create_testing_dataset

### Table Structure

The `stock_daily` table now contains:
- ts_code (Stock code)
- trade_date (Trading date)
- open, high, low, close (OHLC prices)
- pre_close (Previous close price)
- change (Price change)
- pct_chg (Percentage change)
- vol (Volume)
- amount (Trading amount)

### Related Tables

- `stock_daily_adjusted` - Contains adjusted prices with adj_factor
- `adjusted_stock_daily` - Generated from stock_daily with adjustment factors
- `index_daily` - Index data (unchanged)

### SQL Query Examples

Check data coverage:
```sql
SELECT COUNT(DISTINCT ts_code) as stocks,
       MIN(trade_date) as earliest,
       MAX(trade_date) as latest,
       COUNT(*) as total_records
FROM stock_daily;
```

Check latest update:
```sql
SELECT MAX(trade_date) as last_update FROM stock_daily;
```

### Next Steps

1. Run the migration script if using PostgreSQL
2. Rebuild the project: `cargo build --release`
3. Continue with normal data ingestion workflows
