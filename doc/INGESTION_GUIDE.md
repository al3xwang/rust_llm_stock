# Data Ingestion Guide

## Prerequisites

### 1. Get Tushare API Token

1. Register at [Tushare Pro](https://tushare.pro/)
2. Get your API token from https://tushare.pro/user/token
3. Update the `.env` file with your token

### 2. Configure Environment

Update `.env` file in the rust_llm_stock directory:

```bash
DATABASE_URL=postgresql://postgres:12341234@127.0.0.1:5432/research?schema=public
TUSHARE_TOKEN=your_actual_tushare_token_here
```

### 3. Ensure Database is Running

```bash
# Check if PostgreSQL is running
psql $DATABASE_URL -c "SELECT 1"
```

## Quick Daily Update

For daily updates (recommended workflow):

```bash
./daily_update.sh
```

This script will:
1. Pull latest daily stock data from Tushare
2. Automatically update adjusted prices for stocks with adjustment events

**Optional**: Specify custom lookback period (default is 7 days):
```bash
./daily_update.sh 30  # Use 30-day lookback for adjustment detection
```

## Detailed Ingestion Steps

### Step 1: Update Stock Basic Information

This fetches the list of all stocks with basic information (name, industry, listing date, etc.)

```bash
./update_stock_data.sh
```

Or manually:
```bash
cargo run --release --bin ingest_tushare_stock
```

**What it does:**
- Fetches all listed (L), delisted (D), and paused (P) stocks
- Updates the `stock_basic` table
- Shows summary by market and industry

**Expected output:**
```
Fetching stock basic data from Tushare...
Processing listed stocks (L)...
Inserted/Updated 5234 stocks
Processing delisted stocks (D)...
Inserted/Updated 1123 stocks
...
Total: 6357 stocks processed
```

### Step 2: Ingest Stock List (Alternative)

Simple stock list ingestion:

```bash
cargo run --release --bin ingest_tushare_stock
```

### Step 3: Ingest Index Data

For market indices (SSE Composite, SZSE Component, etc.):

```bash
cargo run --release --bin ingest_tushare_index
```

### Step 4: Ingest Daily Stock Data

#### Option A: Pull All Historical Data

```bash
cargo run --release --bin pullall-daily
```

**Features:**
- Fetches from earliest listing date
- Handles incremental updates
- Rate limiting (120 calls/minute)
- Automatically resumes from last date

#### Option B: Ingest Specific Date Range

```bash
cargo run --release --bin ingest_tushare_stock_daily
```

Modify the date range in the code or via command-line args.

### Step 5: Create Training Datasets

After ingesting daily data:

```bash
cargo run --release --bin dataset_creator
```

## Database Tables

### stock_basic
Stores basic stock information:
- `ts_code`: Stock code (e.g., "000001.SZ")
- `symbol`: Stock symbol
- `name`: Stock name
- `industry`: Industry classification
- `list_date`: Listing date
- `list_status`: L (Listed), D (Delisted), P (Paused)

### daily
Stores daily OHLCV data:
- `ts_code`: Stock code
- `trade_date`: Trading date (YYYYMMDD)
- `open`, `high`, `low`, `close`: Prices
- `vol`: Volume
- `amount`: Trading amount

### daily_adjusted
Stores adjusted daily data:
- Same as `daily` table
- Plus `adj_factor`: Adjustment factor for splits/dividends

## Troubleshooting

### Error: "TUSHARE_TOKEN not set"
- Update `.env` file with your actual token
- Make sure `.env` is in the rust_llm_stock directory

### Error: "Database connection failed"
- Check if PostgreSQL is running: `pg_isready`
- Verify DATABASE_URL credentials
- Ensure database 'research' exists

### Error: "API rate limit exceeded"
- Tushare free tier: 120 calls/minute
- The scripts include rate limiting
- Wait a few minutes and retry

### Error: "Table does not exist"
- Run migrations first:
  ```bash
  cd ../stock-miner
  sqlx migrate run
  ```

## Monitoring Progress

### Check stock_basic table:
```sql
SELECT market, list_status, COUNT(*) 
FROM stock_basic 
GROUP BY market, list_status;
```

### Check daily data coverage:
```sql
SELECT 
    COUNT(DISTINCT ts_code) as stocks,
    MIN(trade_date) as earliest,
    MAX(trade_date) as latest,
    COUNT(*) as total_records
FROM stock_daily;
```

### Check latest update:
```sql
SELECT MAX(trade_date) as last_update FROM stock_daily;
```

## Adjustment Events & Adjusted Prices

### Understanding Stock Adjustments

Stock adjustments occur when:
- **Stock splits** (e.g., 2-for-1 split)
- **Dividends** are distributed
- **Rights issues** or other corporate actions

When these events happen, historical prices need to be **backward-adjusted** so price charts remain continuous and comparable.

### Initial Full Adjustment (One-time Setup)

Create the `adjusted_stock_daily` table with all historical adjusted prices:

```bash
cargo run --release --bin create_adjusted_daily
```

**Warning**: This processes all ~5,500 stocks and takes significant time. Only run once for initial setup.

### Incremental Adjustment Updates (Daily Workflow)

After pulling daily data, update adjusted prices for stocks with new adjustment events:

```bash
# Update stocks with adjustments in last 7 days (default)
../target/release/update_adjusted_daily

# Or specify custom lookback period
../target/release/update_adjusted_daily 30  # 30-day lookback
```

**How it works:**
1. Detects stocks where `pre_close != previous day's close` (indicates adjustment event)
2. Recalculates ALL historical adjusted prices for those stocks
3. Updates `adjusted_stock_daily` table with new values

### Automated Daily Workflow

Use the combined script:
```bash
./daily_update.sh          # Uses 7-day lookback
./daily_update.sh 14       # Uses 14-day lookback
```

This automatically:
1. Pulls latest stock_daily data
2. Updates adjusted prices for affected stocks

## Performance Tips

1. **Use release builds** for faster ingestion:
   ```bash
   cargo build --release --bin pullall-daily
   ./target/release/pullall-daily
   ```

2. **Run during off-peak hours** to avoid rate limits

3. **Monitor database size**:
   ```sql
   SELECT pg_size_pretty(pg_database_size('research'));
   ```

4. **Create indexes** for better query performance (already in migrations)

## Next Steps After Ingestion

1. **Verify data quality**:
   ```bash
   cargo run --release --bin check_nulls
   ```

2. **Create adjusted daily records**:
   ```bash
   cargo run --release --bin create_adjusted_stock_daily
   ```

3. **Generate training datasets**:
   ```bash
   cargo run --release --bin export_training_data
   ```

4. **Train model with PyTorch**:
   ```bash
   cargo run --release --features pytorch --bin train
   ```
