# Database Migrations

This directory contains SQL migration files for setting up the database schema on test and production servers.

## Overview

The project uses PostgreSQL and requires two main tables:
1. **ml_training_dataset** - Stock market data with technical indicators for training
2. **stock_history** - Historical stock data for inference

## Migration Files

| File | Description |
|------|-------------|
| `001_create_ml_training_dataset.sql` | Creates the main training dataset table with 28+ features |
| `002_create_stock_history.sql` | Creates the stock history table for inference |

## Running Migrations

### On Test Server (10.0.0.14)

```bash
# Default connection to test server
./run_migrations.sh postgres://postgres:12341234@10.0.0.14/research

# Or with custom connection string
./run_migrations.sh postgres://user:password@host/database
```

### On Local Development

```bash
./run_migrations.sh postgres://postgres:12341234@localhost/research
```

## Migration Tracking

The script automatically:
- Creates a `schema_migrations` table to track applied migrations
- Skips migrations that have already been applied
- Records version, description, and timestamp for each migration
- Provides summary of applied/skipped migrations

## Verifying Migrations

After running migrations, verify the schema:

```bash
# Check tables were created
psql -h 10.0.0.14 -U postgres -d research -c '\dt'

# View migration history
psql -h 10.0.0.14 -U postgres -d research -c 'SELECT * FROM schema_migrations;'

# Describe ml_training_dataset table
psql -h 10.0.0.14 -U postgres -d research -c '\d ml_training_dataset'

# Describe stock_history table
psql -h 10.0.0.14 -U postgres -d research -c '\d stock_history'
```

## Adding New Migrations

1. Create a new SQL file with sequential numbering: `003_description.sql`
2. Add migration metadata in comments:
   ```sql
   -- Migration: Short title
   -- Description: What this migration does
   -- Date: YYYY-MM-DD
   ```
3. Write idempotent SQL (use `IF NOT EXISTS`, `IF EXISTS`, etc.)
4. Run `./run_migrations.sh` to apply

## Schema Overview

### ml_training_dataset Table

Primary table for training data with 50+ columns:
- **Identifiers**: ts_code, trade_date
- **OHLCV**: open, high, low, close, volume, amount
- **Temporal**: weekday, week_no, quarter
- **Moving Averages**: ema_5/10/20/30/60, sma_5/10/20
- **MACD**: macd_line/signal (daily/weekly/monthly)
- **Technical**: rsi_14, cci_14, atr, asi, obv
- **Bollinger**: bb_upper/lower/bandwidth
- **Returns**: pct_change, daily_return, volume_ratio
- **Target**: next_day_return

### stock_history Table

Inference table with streamlined columns:
- **Identifiers**: symbol, timestamp
- **OHLCV**: open, high, low, close, volume
- **Temporal**: month, weekday, quarter
- **Indicators**: sma5, sma20, rsi
- **Returns**: daily_return, volume_ratio

## Rollback

Currently manual rollback is required. To rollback:

```bash
# Drop tables
psql -h host -U user -d database -c 'DROP TABLE IF EXISTS ml_training_dataset CASCADE;'
psql -h host -U user -d database -c 'DROP TABLE IF EXISTS stock_history CASCADE;'

# Remove migration records
psql -h host -U user -d database -c "DELETE FROM schema_migrations WHERE version IN ('001_create_ml_training_dataset', '002_create_stock_history');"
```

## Troubleshooting

### Connection Failed
- Verify PostgreSQL is running: `pg_isready -h 10.0.0.14`
- Check credentials and database name
- Ensure network connectivity: `ping 10.0.0.14`

### Migration Already Applied
The script automatically skips already-applied migrations. Check `schema_migrations` table to see what's been applied.

### Permission Denied
Ensure the database user has CREATE TABLE and CREATE INDEX permissions:
```sql
GRANT CREATE ON DATABASE research TO postgres;
```
