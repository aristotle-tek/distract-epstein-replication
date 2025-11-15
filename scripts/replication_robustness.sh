#!/usr/bin/env bash


# Robustness ARDL for N_t_z novelty using daily_fox_epstein_density

#   bash scripts/replication_robustness.sh (rename?)

"""
run with MMD^2 parquet
ALT_DAY_PANEL="data_processed/runs/tv_epstein_daily_fox_density/mmd2/day_panel.parquet" \
ALT_REPORT_PREFIX="tv_density_mmd2" \
ALT_TAG_PREFIX="mmd2_" \
./scripts/replication_robustness.sh

"""


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

DAY_PANEL=${DAY_PANEL:-$ROOT/data_processed/runs/day_panel/tv_epstein_daily_fox_density/day_panel.parquet}
OUTCOME=${OUTCOME:-N_t_z}
Y_LAGS=${Y_LAGS:-7}
E_LAGS=${E_LAGS:-3}
HAC_LAGS=${HAC_LAGS:-7}
EXPOSURE_COLUMN=${EXPOSURE_COLUMN:-daily_fox_epstein_density}
REPORT_PREFIX=${REPORT_PREFIX:-tv_density}
ALT_DAY_PANEL=${ALT_DAY_PANEL:-}
ALT_REPORT_PREFIX=${ALT_REPORT_PREFIX:-${REPORT_PREFIX}_mmd2}
ALT_TAG_PREFIX=${ALT_TAG_PREFIX:-mmd2_}

if [[ ! -f "$DAY_PANEL" ]]; then
  echo "[error] Expected day panel not found at: $DAY_PANEL" >&2
  echo "Run scripts/replication_setup.sh to build the baseline panel before invoking this script." >&2
  exit 1
fi

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

run_ardl() {
  local report_name=$1
  shift
  local panel_path=${CURRENT_DAY_PANEL:-$DAY_PANEL}
  $PYTHON_BIN src/analyze/ardl.py run \
    --day-panel "$panel_path" \
    --outcomes "$OUTCOME" \
    --y-lags "$Y_LAGS" \
    --e-lags "$E_LAGS" \
    --exposure-column "$EXPOSURE_COLUMN" \
    --use-e-z \
    --hac-lags "$HAC_LAGS" \
    --report-name "$report_name" \
    "$@"
}

run_robustness_suite() {
  local suite_label=$1
  local prefix=$2
  local panel_path=$3
  local tag_prefix=$4

  local label=""
  if [[ -n "$suite_label" ]]; then
    label="[$suite_label] "
  fi

  local prev_prefix=$REPORT_PREFIX
  local prev_panel=""
  local had_panel=0
  if [[ -v CURRENT_DAY_PANEL ]]; then
    had_panel=1
    prev_panel=$CURRENT_DAY_PANEL
  fi
  CURRENT_DAY_PANEL="$panel_path"
  REPORT_PREFIX="$prefix"

  step "${label}Placebo leads (E leads 1..3)"
  run_ardl "${prefix}_leads3" --e-leads 3 --tag "${tag_prefix}leads3"

  step "${label}Shorter exposure window (e-lags = 1)"
  run_ardl "${prefix}_elag1" --e-lags 1 --tag "${tag_prefix}elag1"

  step "${label}Longer exposure window (e-lags = 7)"
  run_ardl "${prefix}_elag7" --e-lags 7 --tag "${tag_prefix}elag7"

  if [[ $had_panel -eq 1 ]]; then
    CURRENT_DAY_PANEL="$prev_panel"
  else
    unset -v CURRENT_DAY_PANEL
  fi
  REPORT_PREFIX="$prev_prefix"
}

run_robustness_suite "" "$REPORT_PREFIX" "$DAY_PANEL" ""

if [[ -n "$ALT_DAY_PANEL" ]]; then
  if [[ ! -f "$ALT_DAY_PANEL" ]]; then
    echo "[error] Alternative day panel not found at: $ALT_DAY_PANEL" >&2
    exit 1
  fi
  run_robustness_suite "MMD^2 novelty" "$ALT_REPORT_PREFIX" "$ALT_DAY_PANEL" "$ALT_TAG_PREFIX"
fi

step "Robustness replication complete"
