#!/usr/bin/env bash
# Optimize PostgreSQL server settings for analytics workloads
# Portable: macOS + Linux. Outputs a tuned conf and can apply via ALTER SYSTEM.
# This copy is provided inside rust_llm_stock for convenience.

set -euo pipefail

OUTPUT_FILE="postgresql_optimized.conf"
DB_URL=""
PGDATA=""
APPLY=false
INSTALL_CONF=false
WORKLOAD="analytics"   # analytics|mixed|oltp
CONCURRENCY=""         # optional overrides, e.g., 32
MAX_CONNS=""           # optional override

# Partial allocation overrides
# Use these when Postgres should only consume part of the machine
PG_RAM_GB_OVERRIDE=""      # e.g., 16 means allocate 16GB for tuning
PG_RAM_PERCENT_OVERRIDE="" # e.g., 35 means use 35% of detected RAM
PG_CORES_OVERRIDE=""       # e.g., 8 means tune for 8 cores

log() { echo "[optimize] $*"; }
warn() { echo "[optimize] WARNING: $*" >&2; }
err() { echo "[optimize] ERROR: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Optimize PostgreSQL for workload patterns (default: analytics).

Options:
  --output <file>           Output conf path (default: postgresql_optimized.conf)
  --db-url <url>            Postgres connection URL (for ALTER SYSTEM)
  --apply                   Write via ALTER SYSTEM to postgresql.auto.conf (requires restart)
  --pgdata <dir>            PGDATA directory to install conf.d/optimized.conf
  --install-conf            Place tuned conf in PGDATA/conf.d and include from postgresql.conf
  --workload <type>         analytics|mixed|oltp (defaults to analytics)
  --concurrency <n>         Expected concurrent active queries (optional)
  --max-conns <n>           Override max_connections assumption (optional)
  --pg-ram-gb <n>           Allocate N GB RAM to Postgres tuning (overrides detection)
  --pg-ram-percent <n>      Allocate N percent of system RAM to Postgres tuning
  --pg-cores <n>            Allocate N CPU cores to Postgres tuning (overrides detection)
  -h|--help                 Show help

Examples:
  ./optimize_postgres.sh --output postgresql_optimized.conf
  ./optimize_postgres.sh --db-url "postgresql://user:pass@host/db" --apply
  ./optimize_postgres.sh --pgdata /var/lib/postgresql/data --install-conf
  # Partial allocation (e.g., server also runs ML training)
  ./optimize_postgres.sh --workload analytics --pg-ram-percent 35 --pg-cores 8 --concurrency 16 --output tuned.conf
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_FILE="$2"; shift 2;;
    --db-url) DB_URL="$2"; shift 2;;
    --apply) APPLY=true; shift;;
    --pgdata) PGDATA="$2"; shift 2;;
    --install-conf) INSTALL_CONF=true; shift;;
    --workload) WORKLOAD="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --max-conns) MAX_CONNS="$2"; shift 2;;
    --pg-ram-gb) PG_RAM_GB_OVERRIDE="$2"; shift 2;;
    --pg-ram-percent) PG_RAM_PERCENT_OVERRIDE="$2"; shift 2;;
    --pg-cores) PG_CORES_OVERRIDE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) warn "Unknown argument: $1"; shift;;
  esac
done

# Detect OS
OS=$(uname -s)
# RAM bytes
get_ram_bytes() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n hw.memsize 2>/dev/null || echo 0
  else
    awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo 2>/dev/null || echo 0
  fi
}
# CPU cores
get_cpu_cores() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n hw.ncpu 2>/dev/null || echo 1
  else
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
  fi
}
# SSD detection (best-effort)
get_is_ssd() {
  if [[ "$OS" == "Darwin" ]]; then
    echo 1
  else
    for dev in /sys/block/*/queue/rotational; do
      if [[ -f "$dev" ]]; then
        rot=$(cat "$dev" || echo 1)
        if [[ "$rot" == "0" ]]; then
          echo 1; return
        fi
      fi
    done
    echo 0
  fi
}

bytes_to_mb() { awk -v b="$1" 'BEGIN{printf "%d", b/1024/1024}' ; }
bytes_to_gb() { awk -v b="$1" 'BEGIN{printf "%d", b/1024/1024/1024}' ; }
mb() { echo "$1MB"; }
gb() { echo "$1GB"; }

RAM_BYTES=$(get_ram_bytes)
CORES=$(get_cpu_cores)
IS_SSD=$(get_is_ssd)
if [[ "$RAM_BYTES" -le 0 ]]; then err "Failed to detect system RAM"; fi

RAM_GB=$(bytes_to_gb "$RAM_BYTES")
RAM_MB=$(bytes_to_mb "$RAM_BYTES")
log "Detected: RAM=${RAM_GB}GB, CORES=${CORES}, SSD=$([[ "$IS_SSD" == "1" ]] && echo yes || echo no), workload=${WORKLOAD}"

# Apply partial allocation overrides to derive effective resources for tuning
EFFECTIVE_CORES="$CORES"
if [[ -n "$PG_CORES_OVERRIDE" ]]; then
  EFFECTIVE_CORES="$PG_CORES_OVERRIDE"
fi

EFFECTIVE_RAM_GB="$RAM_GB"
if [[ -n "$PG_RAM_GB_OVERRIDE" ]]; then
  EFFECTIVE_RAM_GB="$PG_RAM_GB_OVERRIDE"
elif [[ -n "$PG_RAM_PERCENT_OVERRIDE" ]]; then
  EFFECTIVE_RAM_GB=$(( RAM_GB * PG_RAM_PERCENT_OVERRIDE / 100 ))
  if (( EFFECTIVE_RAM_GB < 1 )); then EFFECTIVE_RAM_GB=1; fi
fi
EFFECTIVE_RAM_MB=$(( EFFECTIVE_RAM_GB * 1024 ))
log "Allocating to Postgres: RAM=${EFFECTIVE_RAM_GB}GB (of ${RAM_GB}GB), CORES=${EFFECTIVE_CORES} (of ${CORES})"

# Assumptions for planning
if [[ -z "$CONCURRENCY" ]]; then
  CONCURRENCY=$(( EFFECTIVE_CORES * 2 ))
fi
if [[ -z "$MAX_CONNS" ]]; then
  MAX_CONNS=100
fi
log "Assuming concurrency=${CONCURRENCY}, max_connections=${MAX_CONNS}"

# Memory
SHARED_BUFFERS_GB=$(( EFFECTIVE_RAM_GB * 25 / 100 ))
if (( SHARED_BUFFERS_GB < 1 )); then SHARED_BUFFERS_GB=1; fi
if (( SHARED_BUFFERS_GB > 8 )); then SHARED_BUFFERS_GB=8; fi
SHARED_BUFFERS=$(gb "$SHARED_BUFFERS_GB")

EFFECTIVE_CACHE_GB=$(( EFFECTIVE_RAM_GB * 75 / 100 ))
if (( EFFECTIVE_CACHE_GB < SHARED_BUFFERS_GB )); then EFFECTIVE_CACHE_GB=$SHARED_BUFFERS_GB; fi
EFFECTIVE_CACHE_SIZE=$(gb "$EFFECTIVE_CACHE_GB")

MAINT_WORK_MEM_MB=$(( EFFECTIVE_RAM_MB * 10 / 100 ))
if (( MAINT_WORK_MEM_MB > 2048 )); then MAINT_WORK_MEM_MB=2048; fi
MAINT_WORK_MEM=$(mb "$MAINT_WORK_MEM_MB")

WORK_MEM_MB=$(( EFFECTIVE_RAM_MB * 25 / 100 / CONCURRENCY ))
if (( WORK_MEM_MB < 8 )); then WORK_MEM_MB=8; fi
if (( WORK_MEM_MB > 256 )); then WORK_MEM_MB=256; fi
WORK_MEM=$(mb "$WORK_MEM_MB")

WAL_BUFFERS="64MB"
CHECKPOINT_TIMEOUT="15min"
MAX_WAL_SIZE="4GB"
MIN_WAL_SIZE="1GB"
SYNCHRONOUS_COMMIT="off"
WAL_COMPRESSION="on"

DEFAULT_STAT_TARGET=200
SEQ_PAGE_COST=$([[ "$IS_SSD" == "1" ]] && echo 0.9 || echo 1.0)
RANDOM_PAGE_COST=$([[ "$IS_SSD" == "1" ]] && echo 1.1 || echo 1.5)
EFFECTIVE_IO_CONCURRENCY=$([[ "$IS_SSD" == "1" ]] && echo 200 || echo 2)

MAX_PARALLEL_WORKERS=$(( EFFECTIVE_CORES ))
MAX_PARALLEL_WORKERS_PER_GATHER=$(( (EFFECTIVE_CORES+1)/2 ))
PARALLEL_LEADER_PARTICIPATION="on"

JIT="off"

AUTOVAC_VACUUM_COST_LIMIT=2000
AUTOVAC_FREEZE_MAX_AGE=200000000
AUTOVAC_MULTIXACT_FREEZE_MAX_AGE=400000000

SHARED_PRELOAD_LIBS="pg_stat_statements"
PGSS_MAX=10000
PGSS_SAVE="on"

cat >"$OUTPUT_FILE" <<CONF
# Generated by optimize_postgres.sh (${WORKLOAD})
# Memory
shared_buffers = ${SHARED_BUFFERS}
effective_cache_size = ${EFFECTIVE_CACHE_SIZE}
maintenance_work_mem = ${MAINT_WORK_MEM}
work_mem = ${WORK_MEM}

# WAL / Checkpoints
wal_buffers = ${WAL_BUFFERS}
checkpoint_timeout = ${CHECKPOINT_TIMEOUT}
max_wal_size = ${MAX_WAL_SIZE}
min_wal_size = ${MIN_WAL_SIZE}
synchronous_commit = ${SYNCHRONOUS_COMMIT}
wal_compression = ${WAL_COMPRESSION}

# Planner & IO
default_statistics_target = ${DEFAULT_STAT_TARGET}
seq_page_cost = ${SEQ_PAGE_COST}
random_page_cost = ${RANDOM_PAGE_COST}
effective_io_concurrency = ${EFFECTIVE_IO_CONCURRENCY}

# Parallel query
max_parallel_workers = ${MAX_PARALLEL_WORKERS}
max_parallel_workers_per_gather = ${MAX_PARALLEL_WORKERS_PER_GATHER}
parallel_leader_participation = ${PARALLEL_LEADER_PARTICIPATION}

# JIT
jit = ${JIT}

# Autovacuum (global)
autovacuum_vacuum_cost_limit = ${AUTOVAC_VACUUM_COST_LIMIT}

# Monitoring extension (requires restart)
shared_preload_libraries = '${SHARED_PRELOAD_LIBS}'
pg_stat_statements.max = ${PGSS_MAX}
pg_stat_statements.save = ${PGSS_SAVE}
CONF

log "Wrote recommendations to $OUTPUT_FILE"

apply_alter_system() {
  [[ -z "$DB_URL" ]] && err "--db-url required for --apply"
  log "Applying via ALTER SYSTEM (writes postgresql.auto.conf)"
  psql "$DB_URL" -v ON_ERROR_STOP=1 <<SQL
ALTER SYSTEM SET shared_buffers = '${SHARED_BUFFERS}';
ALTER SYSTEM SET effective_cache_size = '${EFFECTIVE_CACHE_SIZE}';
ALTER SYSTEM SET maintenance_work_mem = '${MAINT_WORK_MEM}';
ALTER SYSTEM SET work_mem = '${WORK_MEM}';
ALTER SYSTEM SET wal_buffers = '${WAL_BUFFERS}';
ALTER SYSTEM SET checkpoint_timeout = '${CHECKPOINT_TIMEOUT}';
ALTER SYSTEM SET max_wal_size = '${MAX_WAL_SIZE}';
ALTER SYSTEM SET min_wal_size = '${MIN_WAL_SIZE}';
ALTER SYSTEM SET synchronous_commit = '${SYNCHRONOUS_COMMIT}';
ALTER SYSTEM SET wal_compression = '${WAL_COMPRESSION}';
ALTER SYSTEM SET default_statistics_target = ${DEFAULT_STAT_TARGET};
ALTER SYSTEM SET seq_page_cost = ${SEQ_PAGE_COST};
ALTER SYSTEM SET random_page_cost = ${RANDOM_PAGE_COST};
ALTER SYSTEM SET effective_io_concurrency = ${EFFECTIVE_IO_CONCURRENCY};
ALTER SYSTEM SET max_parallel_workers = ${MAX_PARALLEL_WORKERS};
ALTER SYSTEM SET max_parallel_workers_per_gather = ${MAX_PARALLEL_WORKERS_PER_GATHER};
ALTER SYSTEM SET parallel_leader_participation = '${PARALLEL_LEADER_PARTICIPATION}';
ALTER SYSTEM SET jit = '${JIT}';
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = ${AUTOVAC_VACUUM_COST_LIMIT};
SQL
  log "ALTER SYSTEM complete. Restart PostgreSQL to take effect."
}

install_conf_into_pgdata() {
  [[ -z "$PGDATA" ]] && err "--pgdata required for --install-conf"
  [[ -d "$PGDATA" ]] || err "PGDATA does not exist: $PGDATA"
  mkdir -p "$PGDATA/conf.d"
  cp -f "$OUTPUT_FILE" "$PGDATA/conf.d/optimized.conf"
  log "Installed $OUTPUT_FILE to $PGDATA/conf.d/optimized.conf"

  POSTGRESQL_CONF="$PGDATA/postgresql.conf"
  if [[ -f "$POSTGRESQL_CONF" ]]; then
    if ! grep -q "include_if_exists = 'conf.d/optimized.conf'" "$POSTGRESQL_CONF"; then
      echo "include_if_exists = 'conf.d/optimized.conf'" >> "$POSTGRESQL_CONF"
      log "Appended include_if_exists to postgresql.conf"
    else
      log "include_if_exists already present"
    fi
  else
    warn "postgresql.conf not found in PGDATA; please add include manually"
  fi
  log "Restart PostgreSQL for changes to apply."
}

if $APPLY; then apply_alter_system; fi
if $INSTALL_CONF; then install_conf_into_pgdata; fi

log "Done. Review $OUTPUT_FILE and apply per your deployment policy."
#!/usr/bin/env bash
# Optimize PostgreSQL server settings for analytics workloads
# Portable: macOS + Linux. Outputs a tuned conf and can apply via ALTER SYSTEM.
# Usage examples:
#   ./optimize_postgres.sh --output postgresql_optimized.conf
#   ./optimize_postgres.sh --db-url "$DATABASE_URL" --apply
#   ./optimize_postgres.sh --pgdata /var/lib/postgresql/data --install-conf

set -euo pipefail

OUTPUT_FILE="postgresql_optimized.conf"
DB_URL=""
PGDATA=""
APPLY=false
INSTALL_CONF=false
WORKLOAD="analytics"   # analytics|mixed|oltp
CONCURRENCY=""         # optional overrides, e.g., 32
MAX_CONNS=""           # optional override

# Partial allocation overrides
# Use these when Postgres should only consume part of the machine
PG_RAM_GB_OVERRIDE=""      # e.g., 16 means allocate 16GB for tuning
PG_RAM_PERCENT_OVERRIDE="" # e.g., 35 means use 35% of detected RAM
PG_CORES_OVERRIDE=""       # e.g., 8 means tune for 8 cores

log() { echo "[optimize] $*"; }
warn() { echo "[optimize] WARNING: $*" >&2; }
err() { echo "[optimize] ERROR: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Optimize PostgreSQL for workload patterns (default: analytics).

Options:
  --output <file>           Output conf path (default: postgresql_optimized.conf)
  --db-url <url>            Postgres connection URL (for ALTER SYSTEM)
  --apply                   Write via ALTER SYSTEM to postgresql.auto.conf (requires restart)
  --pgdata <dir>            PGDATA directory to install conf.d/optimized.conf
  --install-conf            Place tuned conf in PGDATA/conf.d and include from postgresql.conf
  --workload <type>         analytics|mixed|oltp (defaults to analytics)
  --concurrency <n>         Expected concurrent active queries (optional)
  --max-conns <n>           Override max_connections assumption (optional)
  --pg-ram-gb <n>           Allocate N GB RAM to Postgres tuning (overrides detection)
  --pg-ram-percent <n>      Allocate N percent of system RAM to Postgres tuning
  --pg-cores <n>            Allocate N CPU cores to Postgres tuning (overrides detection)
  -h|--help                 Show help

Examples:
  ./optimize_postgres.sh --output postgresql_optimized.conf
  ./optimize_postgres.sh --db-url "postgresql://user:pass@host/db" --apply
  ./optimize_postgres.sh --pgdata /var/lib/postgresql/data --install-conf
  # Partial allocation (e.g., server also runs ML training)
  ./optimize_postgres.sh --workload analytics --pg-ram-percent 35 --pg-cores 8 --concurrency 16 --output tuned.conf
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_FILE="$2"; shift 2;;
    --db-url) DB_URL="$2"; shift 2;;
    --apply) APPLY=true; shift;;
    --pgdata) PGDATA="$2"; shift 2;;
    --install-conf) INSTALL_CONF=true; shift;;
    --workload) WORKLOAD="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --max-conns) MAX_CONNS="$2"; shift 2;;
    --pg-ram-gb) PG_RAM_GB_OVERRIDE="$2"; shift 2;;
    --pg-ram-percent) PG_RAM_PERCENT_OVERRIDE="$2"; shift 2;;
    --pg-cores) PG_CORES_OVERRIDE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) warn "Unknown argument: $1"; shift;;
  esac
done

# Detect OS
OS=$(uname -s)
# RAM bytes
get_ram_bytes() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n hw.memsize 2>/dev/null || echo 0
  else
    awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo 2>/dev/null || echo 0
  fi
}
# CPU cores
get_cpu_cores() {
  if [[ "$OS" == "Darwin" ]]; then
    sysctl -n hw.ncpu 2>/dev/null || echo 1
  else
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
  fi
}
# SSD detection (best-effort)
get_is_ssd() {
  if [[ "$OS" == "Darwin" ]]; then
    # Try to detect main disk rotational; fallback to SSD
    # macOS often uses SSD; assume true if unsure
    echo 1
  else
    # Linux: check rotational flag for major block devices
    for dev in /sys/block/*/queue/rotational; do
      if [[ -f "$dev" ]]; then
        rot=$(cat "$dev" || echo 1)
        if [[ "$rot" == "0" ]]; then
          echo 1; return
        fi
      fi
    done
    echo 0
  fi
}

bytes_to_mb() { awk -v b="$1" 'BEGIN{printf "%d", b/1024/1024}' ; }
bytes_to_gb() { awk -v b="$1" 'BEGIN{printf "%d", b/1024/1024/1024}' ; }
mb() { echo "$1MB"; }
gb() { echo "$1GB"; }

RAM_BYTES=$(get_ram_bytes)
CORES=$(get_cpu_cores)
IS_SSD=$(get_is_ssd)
if [[ "$RAM_BYTES" -le 0 ]]; then err "Failed to detect system RAM"; fi

RAM_GB=$(bytes_to_gb "$RAM_BYTES")
RAM_MB=$(bytes_to_mb "$RAM_BYTES")
log "Detected: RAM=${RAM_GB}GB, CORES=${CORES}, SSD=$([[ "$IS_SSD" == "1" ]] && echo yes || echo no), workload=${WORKLOAD}"

# Apply partial allocation overrides to derive effective resources for tuning
EFFECTIVE_CORES="$CORES"
if [[ -n "$PG_CORES_OVERRIDE" ]]; then
  EFFECTIVE_CORES="$PG_CORES_OVERRIDE"
fi

EFFECTIVE_RAM_GB="$RAM_GB"
if [[ -n "$PG_RAM_GB_OVERRIDE" ]]; then
  EFFECTIVE_RAM_GB="$PG_RAM_GB_OVERRIDE"
elif [[ -n "$PG_RAM_PERCENT_OVERRIDE" ]]; then
  # round down to integer GB
  EFFECTIVE_RAM_GB=$(( RAM_GB * PG_RAM_PERCENT_OVERRIDE / 100 ))
  if (( EFFECTIVE_RAM_GB < 1 )); then EFFECTIVE_RAM_GB=1; fi
fi
EFFECTIVE_RAM_MB=$(( EFFECTIVE_RAM_GB * 1024 ))
log "Allocating to Postgres: RAM=${EFFECTIVE_RAM_GB}GB (of ${RAM_GB}GB), CORES=${EFFECTIVE_CORES} (of ${CORES})"

# Assumptions for planning
if [[ -z "$CONCURRENCY" ]]; then
  # analytics workloads often run few heavy queries; assume 2*effective cores
  CONCURRENCY=$(( EFFECTIVE_CORES * 2 ))
fi
if [[ -z "$MAX_CONNS" ]]; then
  # default assumption if not provided; adjust if your postgresql.conf differs
  MAX_CONNS=100
fi
log "Assuming concurrency=${CONCURRENCY}, max_connections=${MAX_CONNS}"

# Compute recommended settings
# Memory
# shared_buffers ~ 25% of EFFECTIVE RAM (cap at 8GB)
SHARED_BUFFERS_GB=$(( EFFECTIVE_RAM_GB * 25 / 100 ))
if (( SHARED_BUFFERS_GB < 1 )); then SHARED_BUFFERS_GB=1; fi
if (( SHARED_BUFFERS_GB > 8 )); then SHARED_BUFFERS_GB=8; fi
SHARED_BUFFERS=$(gb "$SHARED_BUFFERS_GB")

# effective_cache_size ~ 70-75% of EFFECTIVE RAM
EFFECTIVE_CACHE_GB=$(( EFFECTIVE_RAM_GB * 75 / 100 ))
if (( EFFECTIVE_CACHE_GB < SHARED_BUFFERS_GB )); then EFFECTIVE_CACHE_GB=$SHARED_BUFFERS_GB; fi
EFFECTIVE_CACHE_SIZE=$(gb "$EFFECTIVE_CACHE_GB")

# maintenance_work_mem ~ 10% of EFFECTIVE RAM (cap 2GB)
MAINT_WORK_MEM_MB=$(( EFFECTIVE_RAM_MB * 10 / 100 ))
if (( MAINT_WORK_MEM_MB > 2048 )); then MAINT_WORK_MEM_MB=2048; fi
MAINT_WORK_MEM=$(mb "$MAINT_WORK_MEM_MB")

# work_mem: target ~ (RAM*0.25)/(concurrency). cap 256MB; floor 8MB
WORK_MEM_MB=$(( EFFECTIVE_RAM_MB * 25 / 100 / CONCURRENCY ))
if (( WORK_MEM_MB < 8 )); then WORK_MEM_MB=8; fi
if (( WORK_MEM_MB > 256 )); then WORK_MEM_MB=256; fi
WORK_MEM=$(mb "$WORK_MEM_MB")

# WAL & checkpoints
WAL_BUFFERS="64MB"                    # cap at 64MB (auto if not set)
CHECKPOINT_TIMEOUT="15min"            # fewer, larger checkpoints
MAX_WAL_SIZE="4GB"
MIN_WAL_SIZE="1GB"
SYNCHRONOUS_COMMIT="off"              # analytics batch inserts benefit from off
WAL_COMPRESSION="on"

# Planner/IO
DEFAULT_STAT_TARGET=200
SEQ_PAGE_COST=$([[ "$IS_SSD" == "1" ]] && echo 0.9 || echo 1.0)
RANDOM_PAGE_COST=$([[ "$IS_SSD" == "1" ]] && echo 1.1 || echo 1.5)
EFFECTIVE_IO_CONCURRENCY=$([[ "$IS_SSD" == "1" ]] && echo 200 || echo 2)

# Parallelism (Postgres 11+)
MAX_PARALLEL_WORKERS=$(( EFFECTIVE_CORES ))
MAX_PARALLEL_WORKERS_PER_GATHER=$(( (EFFECTIVE_CORES+1)/2 ))
PARALLEL_LEADER_PARTICIPATION="on"

# JIT: often off for analytics pipelines (less latency variance)
JIT="off"

# Autovacuum tuning (global)
AUTOVAC_VACUUM_COST_LIMIT=2000
AUTOVAC_FREEZE_MAX_AGE=200000000
AUTOVAC_MULTIXACT_FREEZE_MAX_AGE=400000000

# Extensions (requires restart)
SHARED_PRELOAD_LIBS="pg_stat_statements"
PGSS_MAX=10000
PGSS_SAVE="on"

# Generate output conf
cat >"$OUTPUT_FILE" <<CONF
# Generated by optimize_postgres.sh (${WORKLOAD})
# Memory
shared_buffers = ${SHARED_BUFFERS}
effective_cache_size = ${EFFECTIVE_CACHE_SIZE}
maintenance_work_mem = ${MAINT_WORK_MEM}
work_mem = ${WORK_MEM}

# WAL / Checkpoints
wal_buffers = ${WAL_BUFFERS}
checkpoint_timeout = ${CHECKPOINT_TIMEOUT}
max_wal_size = ${MAX_WAL_SIZE}
min_wal_size = ${MIN_WAL_SIZE}
synchronous_commit = ${SYNCHRONOUS_COMMIT}
wal_compression = ${WAL_COMPRESSION}

# Planner & IO
default_statistics_target = ${DEFAULT_STAT_TARGET}
seq_page_cost = ${SEQ_PAGE_COST}
random_page_cost = ${RANDOM_PAGE_COST}
effective_io_concurrency = ${EFFECTIVE_IO_CONCURRENCY}

# Parallel query
max_parallel_workers = ${MAX_PARALLEL_WORKERS}
max_parallel_workers_per_gather = ${MAX_PARALLEL_WORKERS_PER_GATHER}
parallel_leader_participation = ${PARALLEL_LEADER_PARTICIPATION}

# JIT
jit = ${JIT}

# Autovacuum (global)
autovacuum_vacuum_cost_limit = ${AUTOVAC_VACUUM_COST_LIMIT}
# Consider per-table scale factors in a separate SQL (see postgres_table_tuning.sql)

# Monitoring extension (requires restart)
shared_preload_libraries = '${SHARED_PRELOAD_LIBS}'
pg_stat_statements.max = ${PGSS_MAX}
pg_stat_statements.save = ${PGSS_SAVE}

# Notes:
# - Many settings require restart when applied via config files.
# - ALTER SYSTEM will write to postgresql.auto.conf and still require restart.
# - Validate with: SHOW name; after restart.
CONF

log "Wrote recommendations to $OUTPUT_FILE"

apply_alter_system() {
  [[ -z "$DB_URL" ]] && err "--db-url required for --apply"
  log "Applying via ALTER SYSTEM (writes postgresql.auto.conf)"
  psql "$DB_URL" -v ON_ERROR_STOP=1 <<SQL
ALTER SYSTEM SET shared_buffers = '${SHARED_BUFFERS}';
ALTER SYSTEM SET effective_cache_size = '${EFFECTIVE_CACHE_SIZE}';
ALTER SYSTEM SET maintenance_work_mem = '${MAINT_WORK_MEM}';
ALTER SYSTEM SET work_mem = '${WORK_MEM}';
ALTER SYSTEM SET wal_buffers = '${WAL_BUFFERS}';
ALTER SYSTEM SET checkpoint_timeout = '${CHECKPOINT_TIMEOUT}';
ALTER SYSTEM SET max_wal_size = '${MAX_WAL_SIZE}';
ALTER SYSTEM SET min_wal_size = '${MIN_WAL_SIZE}';
ALTER SYSTEM SET synchronous_commit = '${SYNCHRONOUS_COMMIT}';
ALTER SYSTEM SET wal_compression = '${WAL_COMPRESSION}';
ALTER SYSTEM SET default_statistics_target = ${DEFAULT_STAT_TARGET};
ALTER SYSTEM SET seq_page_cost = ${SEQ_PAGE_COST};
ALTER SYSTEM SET random_page_cost = ${RANDOM_PAGE_COST};
ALTER SYSTEM SET effective_io_concurrency = ${EFFECTIVE_IO_CONCURRENCY};
ALTER SYSTEM SET max_parallel_workers = ${MAX_PARALLEL_WORKERS};
ALTER SYSTEM SET max_parallel_workers_per_gather = ${MAX_PARALLEL_WORKERS_PER_GATHER};
ALTER SYSTEM SET parallel_leader_participation = '${PARALLEL_LEADER_PARTICIPATION}';
ALTER SYSTEM SET jit = '${JIT}';
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = ${AUTOVAC_VACUUM_COST_LIMIT};
-- Note: shared_preload_libraries requires restart and careful merge
-- To avoid overwriting existing entries, set manually in postgresql.conf if needed.
SQL
  log "ALTER SYSTEM complete. Restart PostgreSQL to take effect."
}

install_conf_into_pgdata() {
  [[ -z "$PGDATA" ]] && err "--pgdata required for --install-conf"
  [[ -d "$PGDATA" ]] || err "PGDATA does not exist: $PGDATA"
  mkdir -p "$PGDATA/conf.d"
  cp -f "$OUTPUT_FILE" "$PGDATA/conf.d/optimized.conf"
  log "Installed $OUTPUT_FILE to $PGDATA/conf.d/optimized.conf"

  POSTGRESQL_CONF="$PGDATA/postgresql.conf"
  if [[ -f "$POSTGRESQL_CONF" ]]; then
    if ! grep -q "include_if_exists = 'conf.d/optimized.conf'" "$POSTGRESQL_CONF"; then
      echo "include_if_exists = 'conf.d/optimized.conf'" >> "$POSTGRESQL_CONF"
      log "Appended include_if_exists to postgresql.conf"
    else
      log "include_if_exists already present"
    fi
  else
    warn "postgresql.conf not found in PGDATA; please add include manually"
  fi
  log "Restart PostgreSQL for changes to apply."
}

if $APPLY; then apply_alter_system; fi
if $INSTALL_CONF; then install_conf_into_pgdata; fi

log "Done. Review $OUTPUT_FILE and apply per your deployment policy."