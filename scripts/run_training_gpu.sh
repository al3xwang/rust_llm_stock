#!/usr/bin/env bash
set -euo pipefail

# Run dataset creation, export latest 5 years data, and start training on GPU
# Usage: ./scripts/run_training_gpu.sh [history_years] [min_years] [gpus] [loss] [time_decay] [detach] [--huber-delta N]
# Parameters (in order):
#   1) history_years - number of years of history to include when creating the dataset (default: 7)
#   2) min_years     - minimum years of history required for export (0 = include all, default: 0)
#   3) gpus          - GPU id or count to use for training (0 = CPU/no GPU, default: 0)
#   4) loss          - loss function name: 'huber' or 'mse' (default: 'huber')
#   5) time_decay    - float decay parameter passed as TIME_DECAY env var (default: 0.01)
#   6) detach        - 0 or 1: run detached in a GNU Screen session when set to 1 (default: 0)
#   7) huber_delta   - numeric delta value passed to the training binary when loss='huber' (default: 1.0). Can also be set with the named flag `--huber-delta N` which overrides the positional value.
#
# If the final arg 'detach' is set to 1, the entire run will be executed inside a detached
# GNU Screen session. Logs are written to ./logs/run_training_<timestamp>.log
#
# Demo examples:
# 1) Run end-to-end in foreground with defaults (history=7y, include all stocks):
#    ./scripts/run_training_gpu.sh
#
# 2) Run with 5 min years filter and Huber loss on GPU 1, foreground:
#    ./scripts/run_training_gpu.sh 7 5 1 huber 0.01 0 --huber-delta 1.0
#
# 3) Run detached in screen (recommended for long runs):
#    ./scripts/run_training_gpu.sh 7 5 1 huber 0.01 1 --huber-delta 1.0
#    Then attach: screen -r training_<timestamp>
#
# 4) Quick smoke run, no GPU, MSE loss, small history-window:
#    ./scripts/run_training_gpu.sh 3 0 0 mse 0.0
#
# Notes:
# - The script assumes binaries: `dataset_creator`, `export_training_data`, and `train` exist and accept the flags shown.
# - For detached runs a timestamped log is written under ./logs/ and a screen session is created.
# - Environment variables `USE_HUBER_LOSS` and `TIME_DECAY` are exported for the training process.
# - Adjust the TRAIN_CMD block if your training entrypoint expects different flags.


# Defaults
HISTORY_YEARS=${1:-7}
MIN_YEARS=${2:-0}
GPUS=${3:-0}
LOSS=${4:-huber}
TIME_DECAY=${5:-0.01}
DETACH=${6:-0}
HUBER_DELTA=${7:-1.0}

# Allow named override: --huber-delta <value> (overrides positional HUBER_DELTA)
# Example: ./scripts/run_training_gpu.sh 7 5 1 huber 0.01 --huber-delta 1.5
while [ "$#" -gt 0 ]; do
  case "$1" in
    --huber-delta)
      if [ "$#" -ge 2 ]; then
        HUBER_DELTA="$2"
        shift 2
      else
        echo "Error: --huber-delta requires an argument" >&2
        exit 1
      fi
      ;;
    *)
      shift
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

run_steps() {
  echo "==> Running dataset creator (history=${HISTORY_YEARS} years)"
  cargo run --release --bin dataset_creator --  --start-date 20190101 --history-years ${HISTORY_YEARS}

  echo "==> Exporting training data (latest 5 years). Output will be placed in ./data"
  # Ensure data dir exists
  mkdir -p "${REPO_ROOT}/data"
  # Run exporter; pass through --min-years if set (>0)
  if [ "${MIN_YEARS}" -gt 0 ]; then
    cargo run --release --bin export_training_data -- --min-years ${MIN_YEARS}
  else
    cargo run --release --bin export_training_data
  fi

  # Move outputs to data/ (exporter writes train.csv/val.csv/test.csv in cwd)
  if [ -f train.csv ]; then
    mv -f train.csv "${REPO_ROOT}/data/"
  fi
  if [ -f val.csv ]; then
    mv -f val.csv "${REPO_ROOT}/data/"
  fi
  if [ -f test.csv ]; then
    mv -f test.csv "${REPO_ROOT}/data/"
  fi

  echo "==> Starting training on GPU=${GPUS} with loss=${LOSS} and time_decay=${TIME_DECAY}"
  # Set environment flags for training
  if [ "${LOSS}" = "huber" ]; then
    export USE_HUBER_LOSS=1
    echo "⚙️  Exported USE_HUBER_LOSS=1"
  else
    unset USE_HUBER_LOSS 2>/dev/null || true
    echo "⚙️  USE_HUBER_LOSS unset"
  fi
  export TIME_DECAY="${TIME_DECAY}"
  echo "⚙️  Exported TIME_DECAY=${TIME_DECAY}"

  # Training CLI (assumes training entrypoint uses these flags - adjust if different)
  TRAIN_CMD=("cargo" "run" "--release" "features=pytorch" "--bin" "train" "--")
  TRAIN_CMD+=("--gpus" "${GPUS}")
  TRAIN_CMD+=("--loss" "${LOSS}")
  TRAIN_CMD+=("--time-decay" "${TIME_DECAY}")
  if [ "${LOSS}" = "huber" ]; then
    TRAIN_CMD+=("--huber-delta" "${HUBER_DELTA}")
  fi

  echo "Running: ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"

  echo "==> Training finished (or exited)."
}

if [ "${DETACH}" -eq 1 ]; then
  TS=$(date +%Y%m%d-%H%M%S)
  LOG_FILE="${LOG_DIR}/run_training_${TS}.log"
  SESSION="training_${TS}"

  echo "==> Launching detached screen session '${SESSION}' (logs: ${LOG_FILE})"

  TMP_SCRIPT="/tmp/run_training_gpu_${TS}.sh"
  cat > "${TMP_SCRIPT}" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
# Reuse environment variables from outer script
HISTORY_YEARS="${HISTORY_YEARS}"
MIN_YEARS="${MIN_YEARS}"
GPUS="${GPUS}"
LOSS="${LOSS}"
TIME_DECAY="${TIME_DECAY}"

# Print header with environment settings for auditing
echo "===== RUN START: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ====="
echo "REPO_ROOT=${REPO_ROOT}"
echo "HISTORY_YEARS=${HISTORY_YEARS}"
echo "MIN_YEARS=${MIN_YEARS}"
echo "GPUS=${GPUS}"
echo "LOSS=${LOSS}"
echo "TIME_DECAY=${TIME_DECAY}"
echo "HUBER_DELTA=${HUBER_DELTA}"
if [ "${LOSS}" = "huber" ]; then
  echo "USE_HUBER_LOSS=1"
else
  echo "USE_HUBER_LOSS=0"
fi
echo "==============================="

# Execute steps in this temporary script
$(declare -f run_steps)
# Export env vars so training process inherits them
export HUBER_DELTA="${HUBER_DELTA}"
if [ "${LOSS}" = "huber" ]; then
  export USE_HUBER_LOSS=1
else
  unset USE_HUBER_LOSS 2>/dev/null || true
fi
export TIME_DECAY="${TIME_DECAY}"
run_steps
echo "===== RUN END: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ====="
EOS
  chmod +x "${TMP_SCRIPT}"

  # Start the screen session and tee logs
  screen -dmS "${SESSION}" bash -lc "env REPO_ROOT='${REPO_ROOT}' HISTORY_YEARS='${HISTORY_YEARS}' MIN_YEARS='${MIN_YEARS}' GPUS='${GPUS}' LOSS='${LOSS}' TIME_DECAY='${TIME_DECAY}' '${TMP_SCRIPT}' 2>&1 | tee '${LOG_FILE}'"

  echo "Screen session '${SESSION}' started. Attach with: screen -r ${SESSION}"
  echo "Logs: ${LOG_FILE}"
  exit 0
else
  # Run inline (foreground)
  cd "${REPO_ROOT}"
  run_steps
fi