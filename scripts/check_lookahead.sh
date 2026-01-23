#!/usr/bin/env bash
set -euo pipefail

# Basic look-ahead bias checks for exported training CSVs
# - Ensure target columns exist in headers
# - Ensure no suspicious "predicted"/"future" columns are present
# - Ensure per-stock chronological split: train.max < val.min and val.max < test.min

TARGETS=(next_day_return next_day_direction next_3day_return next_3day_direction)

# Resolve CSV_DIR relative to project root (script directory/..), allow override via --data-dir
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_CSV_DIR="$SCRIPT_DIR/../data"
CSV_DIR="$DEFAULT_CSV_DIR"

# Accept optional --data-dir <path>
if [ "${1:-}" = "--data-dir" ] && [ -n "${2:-}" ]; then
  CSV_DIR="$2"
fi

# Find CSV files recursively (support nested folder like data/*/)
files=()
while IFS= read -r -d $'\n' f; do
  files+=("$f")
done < <(find "$CSV_DIR" -maxdepth 2 -type f -name "*.csv" | sort)
if [ ${#files[@]} -eq 0 ]; then
  echo "NOTICE: no CSV files found under $CSV_DIR/ — skipping look-ahead checks (no data committed to repo)."
  echo "Tip: run from repo root with './scripts/check_lookahead.sh' or pass '--data-dir /path/to/data'"
  exit 0
fi

# For convenience, prefer files named train.csv/val.csv/test.csv if present anywhere under CSV_DIR
TRAIN_FILE=$(find "$CSV_DIR" -maxdepth 2 -type f -name "train.csv" | sort | head -n1 || true)
VAL_FILE=$(find "$CSV_DIR" -maxdepth 2 -type f -name "val.csv" | sort | head -n1 || true)
TEST_FILE=$(find "$CSV_DIR" -maxdepth 2 -type f -name "test.csv" | sort | head -n1 || true)

# If named files not found, attempt to infer by file sizes or names
if [ -z "$TRAIN_FILE" ] || [ -z "$VAL_FILE" ] || [ -z "$TEST_FILE" ]; then
  # fall back: try to find files containing 'train'/'val'/'test' in the filename
  for f in "${files[@]}"; do
    fname=$(basename "$f")
    if [ -z "$TRAIN_FILE" ] && echo "$fname" | grep -qi "train"; then TRAIN_FILE="$f"; fi
    if [ -z "$VAL_FILE" ] && echo "$fname" | grep -qi "val"; then VAL_FILE="$f"; fi
    if [ -z "$TEST_FILE" ] && echo "$fname" | grep -qi "test"; then TEST_FILE="$f"; fi
  done
fi

# Debug print of discovered files (helpful when running manually)
echo "Discovered CSV files under $CSV_DIR:"
for f in "${files[@]}"; do echo "  - $f"; done

# Require named splits
if [ -z "$TRAIN_FILE" ] || [ -z "$VAL_FILE" ] || [ -z "$TEST_FILE" ]; then
  echo "ERROR: Could not locate train/val/test CSV files under $CSV_DIR (searched depth=2)."
  echo "Found files:"; for f in "${files[@]}"; do echo "  $f"; done
  exit 5
fi

# Check headers
for f in "$CSV_DIR"/*.csv; do
  [ -f "$f" ] || continue
  header=$(head -n 1 "$f" | tr -d '\r' )
  # Targets presence
  for t in "${TARGETS[@]}"; do
    if ! echo "$header" | grep -q "\b$t\b"; then
      echo "ERROR: target column '$t' missing from header of $f"
      exit 3
    fi
  done
  # Forbidden patterns: predicted|prediction|future (these indicate leakage if present)
  if echo "$header" | grep -Eiq "predicted|prediction|future"; then
    echo "ERROR: suspicious header in $f (contains predicted/prediction/future):"
    echo "  $header"
    exit 4
  fi
done

# Helper to compute per-file per-ts_code min/max date
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for split in train val test; do
  # pick the discovered named file for each split
  case "$split" in
    train) in="$TRAIN_FILE" ; out="$tmpdir/train_ranges.csv" ;;
    val)   in="$VAL_FILE"   ; out="$tmpdir/val_ranges.csv"   ;;
    test)  in="$TEST_FILE"  ; out="$tmpdir/test_ranges.csv"  ;;
  esac

  if [ ! -f "$in" ]; then
    echo "ERROR: expected ${in} to exist (could not locate ${split}.csv under $CSV_DIR)"
    echo "Discovered files:"; for f in "${files[@]}"; do echo "  $f"; done
    exit 5
  fi

  # Extract ts_code and trade_date columns robustly by header discovery
  header_line=$(head -n1 "$in")
  IFS=',' read -r -a cols <<<"$header_line"
  ts_idx=-1; td_idx=-1
  for i in "${!cols[@]}"; do
    c=$(echo "${cols[$i]}" | tr -d '[:space:]' )
    if [ "$c" = "ts_code" ]; then ts_idx=$i; fi
    if [ "$c" = "trade_date" ]; then td_idx=$i; fi
  done
  if [ $ts_idx -lt 0 ] || [ $td_idx -lt 0 ]; then
    echo "ERROR: cannot find ts_code/trade_date header in $in"
    exit 6
  fi
  # awk to compute min/max per ts
  tail -n +2 "$in" | awk -F',' -v ts_col=$((ts_idx+1)) -v td_col=$((td_idx+1)) '
  { gsub(/\r/, "", $td_col); key=$ts_col; d=$td_col; if(!(key in min) || d < min[key]) min[key]=d; if(!(key in max) || d > max[key]) max[key]=d } END { for(k in min) print k","min[k]","max[k] }' > "$out"
done

# Compare ranges: train.max < val.min and val.max < test.min (per-ts if both present)

# Check train.max < val.min
if awk -F',' 'NR==FNR{tmax[$1]=$3; next} { if(($1 in tmax) && tmax[$1] >= $2) { printf("ERROR: Overlap for stock %s: train.max=%s >= val.min=%s\n", $1, tmax[$1], $2); err=1 } } END{ exit err }' "$tmpdir/train_ranges.csv" "$tmpdir/val_ranges.csv"; then
  : # ok
else
  echo "Look-ahead check FAILED"
  exit 7
fi

# Check val.max < test.min
if awk -F',' 'NR==FNR{vmax[$1]=$3; next} { if(($1 in vmax) && vmax[$1] >= $2) { printf("ERROR: Overlap for stock %s: val.max=%s >= test.min=%s\n", $1, vmax[$1], $2); err=1 } } END{ exit err }' "$tmpdir/val_ranges.csv" "$tmpdir/test_ranges.csv"; then
  : # ok
else
  echo "Look-ahead check FAILED"
  exit 7
fi


echo "Look-ahead check PASSED"
exit 0
