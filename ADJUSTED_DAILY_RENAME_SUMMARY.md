# Adjusted Daily Table Rename Summary

**Date**: December 28, 2025

## Overview
Renamed `adjusted_daily` table to `adjusted_stock_daily` across the entire rust_llm_stock codebase to maintain consistent naming convention with `stock_daily`.

## Database Changes

### Table Renamed
- **Old name**: `adjusted_daily`
- **New name**: `adjusted_stock_daily`
- **Records**: 16,037,236 (all preserved)

### Indexes Renamed
1. `idx_adjusted_daily_ts_code` → `idx_adjusted_stock_daily_ts_code`
2. `idx_adjusted_daily_trade_date` → `idx_adjusted_stock_daily_trade_date`
3. `adjusted_daily_pkey` → `adjusted_stock_daily_pkey`

## Code Changes

### Source Files Updated (3 files)
1. **src/bin/create_adjusted_daily.rs** - Multiple changes:
   - Function name: `create_adjusted_daily_table` (kept as is for binary name consistency)
   - All SQL `CREATE TABLE`, `DROP TABLE`, `INSERT INTO`, `CREATE INDEX` statements
   - All query references to the table

2. **src/bin/dataset_creator.rs** - 13 changes:
   - All SQL `SELECT FROM`, `JOIN`, and subquery references
   - Comments referencing the table
   - Query result processing

3. **src/db.rs** - 4 changes:
   - SQL queries fetching adjusted stock data
   - Comments about data source

### Documentation Updated (2 files)
1. **INGESTION_GUIDE.md** - 1 change
2. **TABLE_RENAME_SUMMARY.md** - 3 changes

## Migration Script
Created `rename_adjusted_daily_table.sh` that:
- Checks if `adjusted_daily` exists
- Renames table and all indexes
- Verifies record count preservation
- Displays table structure for confirmation

## Verification

### Build Status
✓ All binaries compile successfully:
- `create_adjusted_daily` - ✓ (warnings only, no errors)
- `dataset_creator` - ✓ (warnings only, no errors)
- `create_testing_dataset` - ✓

### Database Status
✓ Table renamed with all data preserved:
```
Table: adjusted_stock_daily
Records: 16,037,236
Indexes: 3 (primary key + 2 indexes)
```

### Test Query
```sql
-- Verify table accessibility
SELECT COUNT(*) FROM adjusted_stock_daily;
-- Result: 16037236 ✓

-- Verify indexes
\d adjusted_stock_daily
-- Shows all indexes correctly renamed ✓
```

## Command Reference

### Run Migration
```bash
./rename_adjusted_daily_table.sh
```

### Build Binaries
```bash
cargo build --release --bin create_adjusted_daily
cargo build --release --bin dataset_creator
```

### Verify Table
```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM adjusted_stock_daily"
```

## Notes
- Binary name `create_adjusted_daily` kept unchanged for backward compatibility
- All table references in SQL queries updated
- No SQL files in rust_llm_stock needed updates
- Consistent with previous `daily` → `stock_daily` rename pattern
