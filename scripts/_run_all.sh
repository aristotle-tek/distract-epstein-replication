#!/usr/bin/env bash
# Run the complete replication pipeline (need to have embeddings in place first)

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

PYTHON_BIN=${PYTHON:-python}
CONFIG_PATH=${TRCIRCUS_CONFIG:-$ROOT/config.yml}
export TRCIRCUS_CONFIG="$CONFIG_PATH"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT"
else
  case ":$PYTHONPATH:" in
    *":$ROOT:"*) ;;
    *) export PYTHONPATH="$ROOT:$PYTHONPATH" ;;
  esac
fi

TV_DENSITY_PANEL="$ROOT/data_processed/runs/day_panel/tv_epstein_daily_fox_density/day_panel.parquet"
GRTRENDS_PANEL="$ROOT/data_processed/runs/day_panel/gtrends_epstein/day_panel.parquet"
STATIONARITY_SUMMARY="$ROOT/out/reports/stationarity/baseline_stationarity.csv"

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

run_step() {
  local description=$1
  shift
  step "$description"
  "$@"
}

run_step "Replication setup" bash "$ROOT/scripts/replication_setup.sh"
run_step "Baseline ARDL" bash "$ROOT/scripts/replication_ardl.sh"
run_step "ARDL robustness suite" bash "$ROOT/scripts/replication_robustness.sh"
run_step "Exposure robustness" bash "$ROOT/scripts/run_exposure_robustness.sh"
run_step "MMD^2 novelty (with robustness)" env RUN_ROBUSTNESS=1 bash "$ROOT/scripts/replication_mmd2.sh"
run_step "Local projection IRFs" bash "$ROOT/scripts/run_local_projection_irf.sh"
run_step "Falsification (default keyword)" bash "$ROOT/scripts/falsification_tv.sh"
run_step "Falsification (NCAA basketball)" env KEYWORD="ncaabasketball" KEYWORD_SLUG="ncaabasketball" bash "$ROOT/scripts/falsification_tv.sh"
run_step "ARDL using Google Trends Epstein (elag3)" \
  env DAY_PANEL="$GRTRENDS_PANEL" \
      EXPOSURE_COLUMN=Epstein \
      REPORT_NAME=gtrends_epstein_elag3 \
      TAG=gtrends_epstein_elag3 \
      bash "$ROOT/scripts/replication_ardl.sh"

run_step "Stationarity diagnostics" \
  "$PYTHON_BIN" src/analyze/stationarity.py \
    --day-panel "$TV_DENSITY_PANEL" \
    --outcomes N_t_z \
    --exposure-column daily_fox_epstein_density \
    --output "$STATIONARITY_SUMMARY"

run_step "Baseline ARDL (final refresh)" bash "$ROOT/scripts/replication_ardl.sh"
run_step "Compile ARDL result tables" "$PYTHON_BIN" src/compile_ardl_results.py
run_step "Build publication outputs" "$PYTHON_BIN" src/analyze/build_publication_outputs.py

step "Full pipeline complete"
echo "[done] All replication stages finished. Outputs live under out/reports/ and data_processed/runs/."
