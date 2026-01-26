#!/bin/bash
# scripts/run_sliding_train.sh
# Sliding window training/validation split automation for stock prediction
# Usage: ./scripts/run_sliding_train.sh [--window-years N] [--val-months M] [--holdout-months H] [--step-days S] [--steps K] [--run-all]

set -e

# Default parameters
default_window_years=3
default_val_months=6
default_holdout_months=6
default_step_days=7
default_steps=26

today=$(date +%Y-%m-%d)

# Parse arguments
WINDOW_YEARS=$default_window_years
VAL_MONTHS=$default_val_months
HOLDOUT_MONTHS=$default_holdout_months
STEP_DAYS=$default_step_days
STEPS=$default_steps
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --window-years)
      WINDOW_YEARS="$2"; shift 2;;
    --val-months)
      VAL_MONTHS="$2"; shift 2;;
    --holdout-months)
      HOLDOUT_MONTHS="$2"; shift 2;;
    --step-days)
      STEP_DAYS="$2"; shift 2;;
    --steps)
      STEPS="$2"; shift 2;;
    --run-all)
      RUN_ALL=1; shift;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

# Calculate holdout start date (latest HOLDOUT_MONTHS before today)
holdout_start=$(date -v "-${HOLDOUT_MONTHS}m" +%Y-%m-%d)
holdout_end=$today

# Calculate the end date of the first window (ends STEPS*STEP_DAYS before holdout_start)
first_window_end=$(date -j -f "%Y-%m-%d" "$holdout_start" \
  -v "-${STEP_DAYS}d" -v "-$((STEP_DAYS * (STEPS-1)))d" +%Y-%m-%d)

# Main loop
echo "Sliding window training: $STEPS steps, $WINDOW_YEARS years window, $VAL_MONTHS months val, $HOLDOUT_MONTHS months holdout, $STEP_DAYS days/step"
echo "Holdout: $holdout_start to $holdout_end"

for ((i=0; i<$STEPS; i++)); do
  # Calculate window end date for this step
  window_end=$(date -j -f "%Y-%m-%d" "$first_window_end" -v "+$((i*STEP_DAYS))d" +%Y-%m-%d)
  window_start=$(date -j -f "%Y-%m-%d" "$window_end" -v "-${WINDOW_YEARS}y" +%Y-%m-%d)
  val_start=$(date -j -f "%Y-%m-%d" "$window_end" -v "-${VAL_MONTHS}m" +%Y-%m-%d)

  echo "Step $i: $window_start to $window_end (val: $val_start to $window_end)"

  # Export data for this window
  cargo run --release --bin export_training_data -- \
    --start-date "$window_start" \
    --end-date "$window_end" \
    --val-start "$val_start" \
    --val-end "$window_end" \
    --test-cutoff "$holdout_start" \
    --mode sliding

  # Move/rename outputs for this step
  mv data/train.csv data/train_step_$i.csv
  mv data/val.csv data/val_step_$i.csv

done

echo "All $STEPS sliding windows complete."
