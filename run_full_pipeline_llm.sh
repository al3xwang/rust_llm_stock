#!/bin/bash

# Stock Data Ingestion & ML Dataset Pipeline (rust_llm_stock only)
# This script runs all data ingestion steps, then creates adjusted prices, ML dataset, and exports training data to CSV using only rust_llm_stock binaries.
# Usage: ./run_full_pipeline_llm.sh

set -e

LOG=run_full_pipeline_llm_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

WORKDIR="."
cd "$WORKDIR"

# Step 1: Pull stock daily data (rust_llm_stock version)
cargo run --bin pullall-stock-daily --release

# Step 2: Pull index daily data
cargo run --bin pullall-index-daily --release

# Step 3: Pull moneyflow data
cargo run --bin pullall-moneyflow --release
cargo run --bin pullall-moneyflow-ind-dc --release



cargo run --bin pull-dc-index --release
cargo run --bin pull-dc-daily --release
cargo run --bin pull-dc-member --release

# Step 4: Create adjusted daily prices

cargo run --bin create_adjusted_daily --release
cargo run --bin pull_daily_basic --release
cargo run --bin pull-tdx-index --release
cargo run --bin pull-tdx-daily --release
cargo run --bin pull-tdx-member --release

cargo run --bin pullall-ind-dc --release
cargo run --bin pullall-ind-ths --release






# Step 5: Create ML training dataset
cargo run --bin dataset_creator --release

# Step 6: Export training data to CSV
cargo run --bin export_training_data --release

echo "Pipeline completed. See $LOG for details."

# Step 7: Run daily batch prediction

cargo run --release --bin batch_predict --features pytorch -- --use-gpu --concurrency 32