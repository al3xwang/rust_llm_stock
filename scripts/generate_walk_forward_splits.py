import pandas as pd
import os
from datetime import timedelta

# User parameters
DATA_PATH = 'data/training_data.csv'  # Input file
OUT_DIR = 'data/walk_forward_splits'  # Output directory for splits
ORDER_COLS = ['ts_code', 'trade_date']  # Columns to order by
TRAIN_YEARS = 2
TRAIN_MONTHS = 6  # 2.5 years = 2 years + 6 months
VAL_MONTHS = 6
STEP_DAYS = 14  # Changed from 7 to 14 days
DATE_COL = 'trade_date'  # Date column (must be parseable)

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(ORDER_COLS)

# Get unique sorted dates
all_dates = df[DATE_COL].sort_values().unique()

splits = []

# Start from the earliest date
train_start = all_dates[0]
while True:
    # Calculate window ends
    train_end = train_start + pd.DateOffset(years=TRAIN_YEARS, months=TRAIN_MONTHS) - pd.DateOffset(days=1)
    val_start = train_end + pd.DateOffset(days=1)
    val_end = val_start + pd.DateOffset(months=VAL_MONTHS) - pd.DateOffset(days=1)

    # Stop if validation window exceeds available data
    if val_end > all_dates[-1]:
        break

    # Select train and val
    train_mask = (df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)
    val_mask = (df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)
    train_df = df[train_mask]
    val_df = df[val_mask]

    # Skip if not enough data
    if train_df.empty or val_df.empty:
        train_start = train_start + pd.DateOffset(days=STEP_DAYS)
        continue

    # Save splits
    train_file = os.path.join(OUT_DIR, f'train_{train_start.date()}_{train_end.date()}.csv')
    val_file = os.path.join(OUT_DIR, f'val_{val_start.date()}_{val_end.date()}.csv')
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    splits.append((train_file, val_file))

    # Step forward by exactly STEP_DAYS
    train_start = train_start + pd.DateOffset(days=STEP_DAYS)

# Write split list for shell script
with open(os.path.join(OUT_DIR, 'split_list.txt'), 'w') as f:
    for train_file, val_file in splits:
        f.write(f'{train_file},{val_file}\n')

print(f'Generated {len(splits)} walk-forward splits in {OUT_DIR}')
