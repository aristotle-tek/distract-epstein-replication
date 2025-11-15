#!/usr/bin/env bash
# baseline ARDL analyses () N_t_z novelty + the Fox density)

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

PYTHON_BIN=${PYTHON:-python}
CONFIG_PATH=${TRCIRCUS_CONFIG:-$ROOT/config.yml}
export TRCIRCUS_CONFIG="$CONFIG_PATH"

# Ensure project root is importable
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT"
else
  case ":$PYTHONPATH:" in
    *":$ROOT:"*) ;;
    *) export PYTHONPATH="$ROOT:$PYTHONPATH" ;;
  esac
fi

DAY_PANEL=${DAY_PANEL:-$ROOT/data_processed/runs/day_panel/tv_epstein_daily_fox_density/day_panel.parquet}
OUTCOMES=${OUTCOMES:-N_t_z}
Y_LAGS=${Y_LAGS:-7}
E_LAGS=${E_LAGS:-3}
HAC_LAGS=${HAC_LAGS:-7}
EXPOSURE_COLUMN=${EXPOSURE_COLUMN:-daily_fox_epstein_density}
USE_E_Z=${USE_E_Z:-1}
REPORT_NAME=${REPORT_NAME:-baseline_tv_density}
TAG=${TAG:-baseline}
TIMESTAMP=${TIMESTAMP:-0}
NO_CLOBBER=${NO_CLOBBER:-0}
ARDL_EXTRA_ARGS=${ARDL_EXTRA_ARGS:-}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-0}
ROBUSTNESS_OUTCOME=${ROBUSTNESS_OUTCOME:-}

if [[ ! -f "$DAY_PANEL" ]]; then
  echo "[error] Expected day panel not found at: $DAY_PANEL" >&2
  echo "Run scripts/replication_setup.sh (or provide DAY_PANEL) before this script." >&2
  exit 1
fi

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

IFS=',' read -r -a OUTCOME_ARRAY <<< "$OUTCOMES"
if [[ ${#OUTCOME_ARRAY[@]} -eq 0 ]]; then
  echo "[error] OUTCOMES resolved to an empty list" >&2
  exit 1
fi

BASELINE_OUTCOME=${OUTCOME_ARRAY[0]}
if [[ -z "$ROBUSTNESS_OUTCOME" ]]; then
  ROBUSTNESS_OUTCOME="$BASELINE_OUTCOME"
fi

ARDL_CMD=(
  "$PYTHON_BIN" "src/analyze/ardl.py" run
  --day-panel "$DAY_PANEL"
  --outcomes "$OUTCOMES"
  --y-lags "$Y_LAGS"
  --e-lags "$E_LAGS"
  --exposure-column "$EXPOSURE_COLUMN"
  --hac-lags "$HAC_LAGS"
  --report-name "$REPORT_NAME"
  --tag "$TAG"
)

if [[ "$USE_E_Z" == "1" ]]; then
  ARDL_CMD+=(--use-e-z)
fi

if [[ "$TIMESTAMP" == "1" ]]; then
  ARDL_CMD+=(--timestamp)
fi

if [[ "$NO_CLOBBER" == "1" ]]; then
  ARDL_CMD+=(--no-clobber)
fi

if [[ -n "$ARDL_EXTRA_ARGS" ]]; then
  EXTRA_SPLIT=($ARDL_EXTRA_ARGS)
  ARDL_CMD+=("${EXTRA_SPLIT[@]}")
fi

step "Run ARDL baseline"
echo "Command: ${ARDL_CMD[*]}"
"${ARDL_CMD[@]}"

if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
  DEFAULT_PREFIX=${REPORT_NAME##*/}
  ROBUSTNESS_REPORT_PREFIX=${ROBUSTNESS_REPORT_PREFIX:-$DEFAULT_PREFIX}
  step "Run robustness suite"
  (
    export PYTHON="$PYTHON_BIN"
    export TRCIRCUS_CONFIG
    export DAY_PANEL OUTCOME="$ROBUSTNESS_OUTCOME" Y_LAGS E_LAGS HAC_LAGS
    export EXPOSURE_COLUMN="$EXPOSURE_COLUMN"
    export REPORT_PREFIX="$ROBUSTNESS_REPORT_PREFIX"
    bash "$ROOT/scripts/replication_robustness.sh"
  )
fi

step "ARDL replication complete"
