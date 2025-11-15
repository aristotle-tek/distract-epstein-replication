#!/usr/bin/env bash
# Build the MMD^2 novelty series, construct the day-level panel,
# and optionally re-run the robustness suite using the new outcome.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

PYTHON_BIN=${PYTHON:-python}
CONFIG_PATH=${TRCIRCUS_CONFIG:-$ROOT/config.yml}
export TRCIRCUS_CONFIG="$CONFIG_PATH"

# Ensure project root is importable so src.* modules resolve
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT"
else
  case ":$PYTHONPATH:" in
    *":$ROOT:"*) ;;
    *) export PYTHONPATH="$ROOT:$PYTHONPATH" ;;
  esac
fi

WINDOW_START=${WINDOW_START:-2024-10-01}
WINDOW_END=${WINDOW_END:-2025-10-19}
NOVELTY_WINDOW=${NOVELTY_WINDOW:-30}

POSTS_PARQUET=${POSTS_PARQUET:-$ROOT/data_raw/truth_social/truth_posts.parquet}
EMBEDDINGS_PARQUET=${EMBEDDINGS_PARQUET:-$ROOT/data_processed/post_embeddings.parquet}
NOVELTY_MMD2_PARQUET=${NOVELTY_MMD2_PARQUET:-$ROOT/data_processed/novelty_mmd2.parquet}
EPSTEIN_FILE=${EPSTEIN_FILE:-$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv}
EPSTEIN_COLUMN=${EPSTEIN_COLUMN:-daily_fox_epstein_density}
DAY_PANEL_TAG=${DAY_PANEL_TAG:-tv_epstein_daily_fox_density}
DAY_PANEL_COMPONENT=${DAY_PANEL_COMPONENT:-mmd2}
INCLUDE_VOLUME=${INCLUDE_VOLUME:-1}

RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}
ROBUSTNESS_REPORT_BASE=${ROBUSTNESS_REPORT_BASE:-tv_density}
ROBUSTNESS_TAG_PREFIX=${ROBUSTNESS_TAG_PREFIX:-mmd2_}
MMD2_BASELINE_REPORT=${MMD2_BASELINE_REPORT:-tv_density_mmd2_baseline}
MMD2_BASELINE_TAG=${MMD2_BASELINE_TAG:-mmd2_baseline}
MMD2_BASELINE_OUTCOME=${MMD2_BASELINE_OUTCOME:-N_t_z}
MMD2_BASELINE_Y_LAGS=${MMD2_BASELINE_Y_LAGS:-7}
MMD2_BASELINE_E_LAGS=${MMD2_BASELINE_E_LAGS:-3}
MMD2_BASELINE_HAC_LAGS=${MMD2_BASELINE_HAC_LAGS:-7}

DAY_PANEL_RUN_DIR="$ROOT/data_processed/runs/$DAY_PANEL_TAG/$DAY_PANEL_COMPONENT"
DAY_PANEL_PARQUET="$DAY_PANEL_RUN_DIR/day_panel.parquet"

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

step "Validate prerequisite artifacts"
require_file "$POSTS_PARQUET" "posts parquet"
require_file "$EMBEDDINGS_PARQUET" "embeddings parquet"
require_file "$EPSTEIN_FILE" "Epstein exposure file"

step "Compute MMD^2 novelty (window = $NOVELTY_WINDOW)"
"$PYTHON_BIN" src/novelty.py --config "$CONFIG_PATH" build \
  --method mmd2 \
  --emb-path "$EMBEDDINGS_PARQUET" \
  --window "$NOVELTY_WINDOW" \
  --start "$WINDOW_START" \
  --end "$WINDOW_END" \
  --out "$NOVELTY_MMD2_PARQUET"

step "Build day-level panel with MMD^2 novelty"
mkdir -p "$DAY_PANEL_RUN_DIR"
DAY_PANEL_TAG_FULL="$DAY_PANEL_TAG/$DAY_PANEL_COMPONENT"
DAY_PANEL_ARGS=(
  --novelty-parquet "$NOVELTY_MMD2_PARQUET"
  --epstein-file "$EPSTEIN_FILE"
  --epstein-column "$EPSTEIN_COLUMN"
  --start "$WINDOW_START"
  --end "$WINDOW_END"
  --tag "$DAY_PANEL_TAG_FULL"
  --out "$DAY_PANEL_PARQUET"
)
if [[ "$INCLUDE_VOLUME" == "1" ]]; then
  DAY_PANEL_ARGS+=(--posts-parquet "$POSTS_PARQUET")
fi


"$PYTHON_BIN" src/series/day_panel.py --config "$CONFIG_PATH" build "${DAY_PANEL_ARGS[@]}"

if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
  step "Run baseline ARDL for MMD^2 novelty (q=${MMD2_BASELINE_E_LAGS})"
  "$PYTHON_BIN" src/analyze/ardl.py run \
    --day-panel "$DAY_PANEL_PARQUET" \
    --outcomes "$MMD2_BASELINE_OUTCOME" \
    --y-lags "$MMD2_BASELINE_Y_LAGS" \
    --e-lags "$MMD2_BASELINE_E_LAGS" \
    --exposure-column "$EPSTEIN_COLUMN" \
    --use-e-z \
    --hac-lags "$MMD2_BASELINE_HAC_LAGS" \
    --report-name "$MMD2_BASELINE_REPORT" \
    --tag "$MMD2_BASELINE_TAG"

  step "Run robustness suite for MMD^2 novelty"
  ALT_DAY_PANEL="$DAY_PANEL_PARQUET" \
    ALT_REPORT_PREFIX="${ROBUSTNESS_REPORT_BASE}_mmd2" \
    ALT_TAG_PREFIX="$ROBUSTNESS_TAG_PREFIX" \
    ./scripts/replication_robustness.sh
else
  echo "[info] Robustness run skipped (set RUN_ROBUSTNESS=1 to enable)." >&2
fi

echo
echo "MMD^2 day panel written to: $DAY_PANEL_PARQUET"
if [[ "$RUN_ROBUSTNESS" != "1" ]]; then
  echo "Next step: optionally run robustness with RUN_ROBUSTNESS=1 or manually via:" >&2
  echo "  ALT_DAY_PANEL=\"$DAY_PANEL_PARQUET\" \\" >&2
  echo "  ALT_REPORT_PREFIX=\"${ROBUSTNESS_REPORT_BASE}_mmd2\" \\" >&2
  echo "  ALT_TAG_PREFIX=\"$ROBUSTNESS_TAG_PREFIX\" \\" >&2
  echo "  ./scripts/replication_robustness.sh" >&2
fi
