#!/bin/bash
# Memory-Optimized Batch Predict - Quick Reference
# Created: 2026-01-18

# ============================================================================
# RECOMMENDED SETTINGS BY SERVER CAPACITY
# ============================================================================

# LOW MEMORY (12GB PostgreSQL allocation, 19GB for other workloads)
# Use this when running ML training + batch_predict simultaneously
low_memory() {
    cargo run --release --bin batch_predict -- \
        --batch-size 25 \
        --concurrency 8 \
        --lookback-days 120 \
        --use-gpu \
        "$@"
}

# MEDIUM MEMORY (12GB PostgreSQL, batch_predict only)
# Default balanced settings
medium_memory() {
    cargo run --release --bin batch_predict -- \
        --batch-size 40 \
        --concurrency 12 \
        --lookback-days 180 \
        --use-gpu \
        "$@"
}

# HIGH MEMORY (Dedicated prediction server, 24GB+ RAM)
# Maximum throughput
high_memory() {
    cargo run --release --bin batch_predict -- \
        --batch-size 100 \
        --concurrency 20 \
        --lookback-days 400 \
        --use-gpu \
        "$@"
}

# TESTING MODE (Minimal memory for debugging)
testing() {
    cargo run --release --bin batch_predict -- \
        --batch-size 10 \
        --concurrency 2 \
        --lookback-days 90 \
        --limit 50 \
        "$@"
}

# ============================================================================
# CHUNKED PROCESSING (For OOM situations)
# ============================================================================

# Process 5000 stocks in 10 chunks of 500
chunked_process() {
    for i in {0..9}; do
        echo "Processing chunk $((i+1))/10 (stocks $((i*500)) to $((i*500+499)))"
        cargo run --release --bin batch_predict -- \
            --offset $((i * 500)) \
            --limit 500 \
            --batch-size 25 \
            --concurrency 8 \
            --lookback-days 120 \
            --use-gpu \
            "$@"
        
        echo "Chunk $((i+1)) complete. Waiting 30s for memory cleanup..."
        sleep 30
    done
}

# ============================================================================
# MONITORING HELPERS
# ============================================================================

# Watch memory usage during batch_predict
watch_memory() {
    ssh alex@10.0.0.12 'watch -n 2 "free -h && echo && ps aux | grep batch_predict | grep -v grep"'
}

# Check if OOM killer triggered
check_oom() {
    ssh alex@10.0.0.12 'sudo dmesg | tail -100 | grep -i "out of memory"'
}

# Show PostgreSQL memory stats
pg_memory() {
    ssh alex@10.0.0.12 'cd /home/alex/rust_llm_stock && set -a && . ./.env && set +a && psql "$DATABASE_URL" -c "SHOW shared_buffers; SHOW work_mem; SHOW effective_cache_size;"'
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

show_usage() {
    cat << EOF
Memory-Optimized Batch Predict - Quick Reference

USAGE:
  source memory_tuning_quick_ref.sh
  <function_name> [additional args]

FUNCTIONS:

  low_memory        - 12GB PostgreSQL + concurrent workloads (25 stocks/batch)
  medium_memory     - 12GB PostgreSQL, dedicated prediction (40 stocks/batch)
  high_memory       - 24GB+ RAM, maximum throughput (100 stocks/batch)
  testing           - Minimal memory for debugging (10 stocks/batch)
  chunked_process   - Process in 500-stock chunks for OOM recovery

  watch_memory      - Monitor live memory usage
  check_oom         - Check if OOM killer was triggered
  pg_memory         - Show PostgreSQL memory configuration

EXAMPLES:

  # Low memory mode with date range
  low_memory --start-date 20260101 --end-date 20260118

  # High memory mode, process only 100 stocks
  high_memory --limit 100

  # Chunked processing for OOM recovery
  chunked_process --start-date 20260101

  # Test run with 10 stocks
  testing

MONITORING:

  # Check memory before starting
  ssh alex@10.0.0.12 'free -h'

  # Watch memory during run (in separate terminal)
  watch_memory

  # Check for OOM kills after failure
  check_oom

PARAMETERS:

  --batch-size N        Stocks per GPU batch (default varies by mode)
  --concurrency N       Concurrent DB fetches (default varies by mode)
  --lookback-days N     Historical data window (120/180/400)
  --start-date YYYYMMDD Start date for predictions
  --end-date YYYYMMDD   End date for predictions
  --limit N             Process only first N stocks (for testing)
  --offset N            Skip first N stocks (for parallel processing)
  --use-gpu             Enable GPU acceleration

MEMORY ESTIMATES:

  Low Memory Mode:     ~200 MB peak (25 × 8 batches)
  Medium Memory Mode:  ~480 MB peak (40 × 12 batches)
  High Memory Mode:    ~2 GB peak (100 × 20 batches)
  Testing Mode:        ~20 MB peak (10 × 2 batches)

TROUBLESHOOTING:

  If process killed (OOM):
    1. check_oom                    # Confirm OOM kill
    2. chunked_process              # Process in smaller chunks
    3. Reduce --lookback-days 90    # Minimum for indicators

  If "Batch memory usage" warnings:
    1. Reduce --batch-size by 50%
    2. Reduce --concurrency by 50%
    3. Monitor with watch_memory

  If very slow:
    1. Check pg_memory              # Ensure PostgreSQL tuned
    2. Increase --concurrency       # More parallel fetches
    3. Use --use-gpu                # GPU acceleration

EOF
}

# ============================================================================
# AUTO-DETECT AVAILABLE MEMORY AND SUGGEST MODE
# ============================================================================

suggest_mode() {
    local total_ram=$(ssh alex@10.0.0.12 'free -g | grep Mem | awk "{print \$2}"')
    local pg_ram=12  # PostgreSQL allocation
    local available=$((total_ram - pg_ram))
    
    echo "Server RAM: ${total_ram}GB total, ${pg_ram}GB for PostgreSQL, ${available}GB available"
    echo ""
    
    if [ $available -lt 8 ]; then
        echo "⚠️  LOW MEMORY: Recommend low_memory mode (25 stocks/batch)"
        echo "   Use: low_memory --start-date 20260101"
    elif [ $available -lt 16 ]; then
        echo "✓ MEDIUM MEMORY: Recommend medium_memory mode (40 stocks/batch)"
        echo "   Use: medium_memory --start-date 20260101"
    else
        echo "✓ HIGH MEMORY: Can use high_memory mode (100 stocks/batch)"
        echo "   Use: high_memory --start-date 20260101"
    fi
}

# ============================================================================
# RUN ON SOURCE
# ============================================================================

if [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
elif [ "$1" = "suggest" ]; then
    suggest_mode
fi
