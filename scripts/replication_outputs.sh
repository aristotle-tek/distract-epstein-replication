#!/usr/bin/env bash
# Generate tables and figures after running the rest

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

ARGS=()
if [[ -n "${COMPILED_RESULTS:-}" ]]; then
  ARGS+=(--compiled-results "$COMPILED_RESULTS")
fi
if [[ -n "${REPORTS_DIR:-}" ]]; then
  ARGS+=(--reports-dir "$REPORTS_DIR")
fi

echo "[info] Building manuscript tables and figures"
"$PYTHON_BIN" src/analyze/build_publication_outputs.py "${ARGS[@]}"

echo "[done] Outputs written to $(python - <<'PY'
from src.utils import paths
print(paths.figures_dir())
print(paths.tables_dir())
PY
)"
