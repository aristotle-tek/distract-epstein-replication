#!/usr/bin/env bash
# Run local projection IRF estimation on a prepared day-level panel.

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

DAY_PANEL=${DAY_PANEL:-$ROOT/data_processed/runs/day_panel/tv_epstein_daily_fox_density/day_panel.parquet}
OUTCOMES=${OUTCOMES:-N_t_z}
EXPOSURE_COLUMN=${EXPOSURE_COLUMN:-daily_fox_epstein_density}
USE_E_Z=${USE_E_Z:-1}
Y_LAGS=${Y_LAGS:-7}
LEADS=${LEADS:-5}
LAGS=${LAGS:-14}
HAC_LAGS=${HAC_LAGS:-7}
INCLUDE_WEEKDAY=${INCLUDE_WEEKDAY:-1}
INCLUDE_MONTH=${INCLUDE_MONTH:-1}
REPORT_NAME=${REPORT_NAME:-lp_irf_baseline}
TIMESTAMP=${TIMESTAMP:-0}
NO_CLOBBER=${NO_CLOBBER:-0}
OUTDIR=${OUTDIR:-}
TAG=${TAG:-}
LP_EXTRA_ARGS=${LP_EXTRA_ARGS:-}

if [[ ! -f "$DAY_PANEL" ]]; then
  echo "[error] Expected day panel not found at: $DAY_PANEL" >&2
  exit 1
fi

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

CMD=(
  "$PYTHON_BIN" "src/analyze/local_projection_irf.py" run
  --day-panel "$DAY_PANEL"
  --outcomes "$OUTCOMES"
  --exposure-column "$EXPOSURE_COLUMN"
  --y-lags "$Y_LAGS"
  --leads "$LEADS"
  --lags "$LAGS"
  --hac-lags "$HAC_LAGS"
  --report-name "$REPORT_NAME"
)

if [[ "$USE_E_Z" != "1" ]]; then
  CMD+=(--no-use-e-z)
fi

if [[ "$INCLUDE_WEEKDAY" != "1" ]]; then
  CMD+=(--no-include-weekday)
fi

if [[ "$INCLUDE_MONTH" != "1" ]]; then
  CMD+=(--no-include-month)
fi

if [[ "$TIMESTAMP" == "1" ]]; then
  CMD+=(--timestamp)
fi

if [[ "$NO_CLOBBER" == "1" ]]; then
  CMD+=(--no-clobber)
fi

if [[ -n "$OUTDIR" ]]; then
  CMD+=(--outdir "$OUTDIR")
fi

if [[ -n "$TAG" ]]; then
  CMD+=(--tag "$TAG")
fi

if [[ -n "$LP_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=($LP_EXTRA_ARGS)
  CMD+=("${EXTRA_SPLIT[@]}")
fi

step "Run local projection IRF"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

step "Local projection IRF complete"
