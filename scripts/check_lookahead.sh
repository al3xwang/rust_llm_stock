#!/usr/bin/env bash
set -euo pipefail

# Basic look-ahead bias checks for exported training CSVs
# - Ensure target columns exist in headers
# - Ensure no suspicious "predicted"/"future" columns are present
# - Ensure per-stock chronological split: train.max < val.min and val.max < test.min

TARGETS=(next_day_return next_day_direction next_3day_return next_3day_direction)
CSV_DIR="data"

shopt -s nullglob
files=("$CSV_DIR"/*.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "NOTICE: no CSV files found under $CSV_DIR/ — skipping look-ahead checks (no data committed to repo)."
  exit 0
fi
shopt -u nullglob

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
  in="$CSV_DIR/${split}.csv"
  out="$tmpdir/${split}_ranges.csv"
  if [ ! -f "$in" ]; then
    echo "ERROR: expected ${in} to exist"
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

fail=0
# Build associative arrays from files
declare -A TRAIN_MIN TRAIN_MAX VAL_MIN VAL_MAX TEST_MIN TEST_MAX
while IFS=',' read -r k mn mx; do TRAIN_MIN[$k]=$mn; TRAIN_MAX[$k]=$mx; done < "$tmpdir/train_ranges.csv" || true
while IFS=',' read -r k mn mx; do VAL_MIN[$k]=$mn; VAL_MAX[$k]=$mx; done < "$tmpdir/val_ranges.csv" || true
while IFS=',' read -r k mn mx; do TEST_MIN[$k]=$mn; TEST_MAX[$k]=$mx; done < "$tmpdir/test_ranges.csv" || true

# Check each ts present in both train and val
for k in "${!TRAIN_MAX[@]}"; do
  if [ -n "${VAL_MIN[$k]:-}" ]; then
    if [[ "${TRAIN_MAX[$k]}" >= "${VAL_MIN[$k]}" ]]; then
      echo "ERROR: Overlap for stock $k: train.max=${TRAIN_MAX[$k]} >= val.min=${VAL_MIN[$k]}"
      fail=1
    fi
  fi
  if [ -n "${TEST_MIN[$k]:-}" ] && [ -n "${VAL_MAX[$k]:-}" ]; then
    if [[ "${VAL_MAX[$k]}" >= "${TEST_MIN[$k]}" ]]; then
      echo "ERROR: Overlap for stock $k: val.max=${VAL_MAX[$k]} >= test.min=${TEST_MIN[$k]}"
      fail=1
    fi
  fi
done

if [ $fail -ne 0 ]; then
  echo "Look-ahead check FAILED"
  exit 7
fi

echo "Look-ahead check PASSED"
exit 0
