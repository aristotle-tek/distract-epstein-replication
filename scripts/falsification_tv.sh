#!/usr/bin/env bash
# Run the falsification pipeline for alternative TV transcript keywords
#
#   1. Merge per-network transcript csvs into a combined exposure file.
#   2. Build the day-level panel with the selected exposure column.
#   3. Run the ARDL analysis

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

PYTHON_BIN=${PYTHON:-python}
CONFIG_PATH=${TRCIRCUS_CONFIG:-$ROOT/config.yml}
export TRCIRCUS_CONFIG="$CONFIG_PATH"

# Ensure project root is on PYTHONPATH for src.* imports
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT"
else
  case ":$PYTHONPATH:" in
    *":$ROOT:"*) ;;
    *) export PYTHONPATH="$ROOT:$PYTHONPATH" ;;
  esac
fi

KEYWORD=${KEYWORD:-taylorswift}
SLUG=${KEYWORD_SLUG:-$($PYTHON_BIN - "$KEYWORD" <<'PY'
import re, sys
value = sys.argv[1]
value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
value = re.sub(r"_+", "_", value).strip("_")
print(value or "series")
PY
)}

FOX_DAILY=${FOX_DAILY:-$ROOT/data_processed/tv_transcripts/${SLUG}/daily_fox_${SLUG}.csv}
CNN_DAILY=${CNN_DAILY:-$ROOT/data_processed/tv_transcripts/${SLUG}/daily_cnn_${SLUG}.csv}
MSNBC_DAILY=${MSNBC_DAILY:-$ROOT/data_processed/tv_transcripts/${SLUG}/daily_msnbc_${SLUG}.csv}
MERGED_OUTPUT=${MERGED_OUTPUT:-$ROOT/data_processed/tv_transcripts/falsification/${SLUG}/daily_mg_${SLUG}.csv}

EXPOSURE_COLUMN=${EXPOSURE_COLUMN:-daily_fox_${SLUG}_density}
DAY_PANEL_TAG=${DAY_PANEL_TAG:-tv_${SLUG}_daily_fox_density}
DAY_PANEL_COMPONENT=${DAY_PANEL_COMPONENT:-mmd2}
ARDL_REPORT_NAME=${ARDL_REPORT_NAME:-falsification_${SLUG}}
ARDL_TAG=${ARDL_TAG:-falsification_${SLUG}}
OUTCOMES=${OUTCOMES:-N_t_z}
NO_CLOBBER=${NO_CLOBBER:-1}
TIMESTAMP=${TIMESTAMP:-0}

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

require_file() {
  local path=$1
  local description=$2
  if [[ ! -e "$path" ]]; then
    echo "[error] Missing $description at: $path" >&2
    exit 1
  fi
}

step "Merge daily transcript CSVs (${KEYWORD} → ${SLUG})"
mkdir -p "$(dirname "$MERGED_OUTPUT")"
MERGE_ARGS=(
  merge-tv
  --label "$KEYWORD"
  --slug "$SLUG"
  --out "$MERGED_OUTPUT"
)
if [[ -f "$FOX_DAILY" ]]; then
  MERGE_ARGS+=(--fox "$FOX_DAILY")
fi
if [[ -f "$CNN_DAILY" ]]; then
  MERGE_ARGS+=(--cnn "$CNN_DAILY")
fi
if [[ -f "$MSNBC_DAILY" ]]; then
  MERGE_ARGS+=(--msnbc "$MSNBC_DAILY")
fi

if [[ ${#MERGE_ARGS[@]} -le 4 ]]; then
  echo "[error] No network CSVs found for merging." >&2
  echo "Provide FOX_DAILY, CNN_DAILY, or MSNBC_DAILY." >&2
  exit 1
fi

"$PYTHON_BIN" src/data_loaders.py "${MERGE_ARGS[@]}"
require_file "$MERGED_OUTPUT" "merged transcript CSV"

step "Build day panel for '${SLUG}'"
export EPSTEIN_FILE="$MERGED_OUTPUT"
export EPSTEIN_COLUMN="$EXPOSURE_COLUMN"
export DAY_PANEL_TAG
export DAY_PANEL_COMPONENT
"$ROOT/scripts/replication_mmd2.sh"

DAY_PANEL_PATH="$ROOT/data_processed/runs/$DAY_PANEL_TAG/$DAY_PANEL_COMPONENT/day_panel.parquet"
require_file "$DAY_PANEL_PATH" "day panel parquet"

step "Run ARDL falsification model"
export DAY_PANEL="$DAY_PANEL_PATH"
export OUTCOMES
export EXPOSURE_COLUMN
export REPORT_NAME="$ARDL_REPORT_NAME"
export TAG="$ARDL_TAG"
export NO_CLOBBER
export TIMESTAMP
"$ROOT/scripts/replication_ardl.sh"

FIRST_OUTCOME=${OUTCOMES%%,*}
RESULT_CSV="$ROOT/out/reports/$ARDL_REPORT_NAME/ardl_${FIRST_OUTCOME}.csv"
if [[ -f "$RESULT_CSV" ]]; then
  echo "\n[done] Primary ARDL results available at: $RESULT_CSV"
else
  echo "\n[warn] Expected ARDL results CSV not found at: $RESULT_CSV" >&2
fi
