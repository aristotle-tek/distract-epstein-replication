#!/usr/bin/env bash
# Build day panels and run ARDL robustness checks across multiple exposure measures

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

NOVELTY_PARQUET=${NOVELTY_PARQUET:-$ROOT/data_processed/novelty_energy.parquet}
POSTS_PARQUET=${POSTS_PARQUET:-$ROOT/data_raw/truth_social/truth_posts.parquet}
BUCKET_PARQUET=${BUCKET_PARQUET:-}
START_DATE=${START_DATE:-2024-10-01}
END_DATE=${END_DATE:-2025-10-19}
OUTCOMES=${OUTCOMES:-N_t_z}
Y_LAGS=${Y_LAGS:-7}
E_LAGS=${E_LAGS:-3}
HAC_LAGS=${HAC_LAGS:-7}
INCLUDE_WEEKDAY=${INCLUDE_WEEKDAY:-1}
INCLUDE_MONTH=${INCLUDE_MONTH:-1}
NO_CLOBBER=${NO_CLOBBER:-0}
ARDL_EXTRA_ARGS=${ARDL_EXTRA_ARGS:-}
SUMMARY_CSV=${SUMMARY_CSV:-$ROOT/data_processed/runs/ardl_robustness/robustness_summary.csv}

if [[ ! -f "$NOVELTY_PARQUET" ]]; then
  echo "[error] Novelty parquet not found: $NOVELTY_PARQUET" >&2
  exit 1
fi

if [[ ! -f "$POSTS_PARQUET" ]]; then
  echo "[error] Posts parquet not found: $POSTS_PARQUET" >&2
  exit 1
fi

if [[ -n "$BUCKET_PARQUET" && ! -f "$BUCKET_PARQUET" ]]; then
  echo "[warn] Bucket parquet not found at $BUCKET_PARQUET; proceeding without it." >&2
  BUCKET_PARQUET=""
fi

DEFAULT_SPECS=(
  "tag=tv_epstein_daily_fox_epstein_density epstein_file=$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv epstein_column=daily_fox_epstein_density exposure_column=daily_fox_epstein_density use_e_z=1"
  "tag=tv_epstein_daily_fox_epstein_hits epstein_file=$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv epstein_column=daily_fox_epstein_hits exposure_column=daily_fox_epstein_hits use_e_z=1"
  "tag=tv_epstein_daily_cnn_epstein_density epstein_file=$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv epstein_column=daily_cnn_epstein_density exposure_column=daily_cnn_epstein_density use_e_z=1"
  "tag=tv_epstein_daily_msnbc_epstein_density epstein_file=$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv epstein_column=daily_msnbc_epstein_density exposure_column=daily_msnbc_epstein_density use_e_z=1"
  "tag=tv_epstein_daily_all3_epstein_mean epstein_file=$ROOT/data_processed/tv_transcripts/daily_mg_epstein.csv epstein_column=daily_all3_epstein_mean exposure_column=daily_all3_epstein_mean use_e_z=1"
  "tag=gtrends_epstein epstein_file=$ROOT/data_raw/gtrends/merged__epstein__US__2024-11-01__2025-10-19.csv epstein_column=Epstein exposure_column=Epstein use_e_z=1"
)

if [[ -n "${ROBUSTNESS_SPECS:-}" ]]; then
  IFS=';' read -r -a SPEC_LIST <<< "$ROBUSTNESS_SPECS"
else
  SPEC_LIST=("${DEFAULT_SPECS[@]}")
fi

if [[ ${#SPEC_LIST[@]} -eq 0 ]]; then
  echo "[error] No robustness specifications supplied." >&2
  exit 1
fi

step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

SUMMARY_ENTRIES=()

for spec in "${SPEC_LIST[@]}"; do
  declare -A S=()
  for pair in $spec; do
    key=${pair%%=*}
    value=${pair#*=}
    S[$key]=$value
  done

  TAG=${S[tag]:-}
  EPSTEIN_FILE=${S[epstein_file]:-}
  EPSTEIN_COLUMN=${S[epstein_column]:-}
  if [[ -z "$TAG" || -z "$EPSTEIN_FILE" || -z "$EPSTEIN_COLUMN" ]]; then
    echo "[error] Spec is missing tag/epstein_file/epstein_column: $spec" >&2
    exit 1
  fi

  if [[ ! -f "$EPSTEIN_FILE" ]]; then
    echo "[error] Exposure source not found: $EPSTEIN_FILE (spec: $TAG)" >&2
    exit 1
  fi

  EXPOSURE_COLUMN=${S[exposure_column]:-$EPSTEIN_COLUMN}
  USE_E_Z=${S[use_e_z]:-1}
  REPORT_NAME=${S[report_name]:-$TAG}
  ARDL_SPEC_EXTRA=${S[ardl_extra_args]:-}

  DAY_PANEL_OUT=${S[day_panel_out]:-$ROOT/data_processed/runs/day_panel/$TAG/day_panel.parquet}
  ARDL_OUTDIR=${S[outdir]:-$ROOT/data_processed/runs/ardl_robustness/$TAG}

  mkdir -p "$(dirname "$DAY_PANEL_OUT")"
  mkdir -p "$ARDL_OUTDIR"

  if [[ "$NO_CLOBBER" == "1" && -f "$DAY_PANEL_OUT" ]]; then
    echo "[info] Skipping day panel build for $TAG (exists and NO_CLOBBER=1)."
  else
    step "Build day panel: $TAG"
    DAY_PANEL_CMD=(
      "$PYTHON_BIN" src/series/day_panel.py --config "$CONFIG_PATH" build
      --novelty-parquet "$NOVELTY_PARQUET"
      --epstein-file "$EPSTEIN_FILE"
      --epstein-column "$EPSTEIN_COLUMN"
      --posts-parquet "$POSTS_PARQUET"
      --start "$START_DATE"
      --end "$END_DATE"
      --tag "$TAG"
      --out "$DAY_PANEL_OUT"
    )
    if [[ -n "$BUCKET_PARQUET" ]]; then
      DAY_PANEL_CMD+=(--bucket-parquet "$BUCKET_PARQUET")
    fi
    echo "Command: ${DAY_PANEL_CMD[*]}"
    "${DAY_PANEL_CMD[@]}"
  fi

  if [[ ! -f "$DAY_PANEL_OUT" ]]; then
    echo "[error] Day panel missing after build: $DAY_PANEL_OUT" >&2
    exit 1
  fi

  SUMMARY_SUFFIX="ardl_summary_${TAG}.json"

  if [[ "$NO_CLOBBER" == "1" ]]; then
    ARDL_EXTRA_FLAG=(--no-clobber)
  else
    ARDL_EXTRA_FLAG=()
  fi

  step "Run ARDL: $TAG"
  ARDL_CMD=(
    "$PYTHON_BIN" src/analyze/ardl.py run
    --day-panel "$DAY_PANEL_OUT"
    --outcomes "$OUTCOMES"
    --y-lags "$Y_LAGS"
    --e-lags "$E_LAGS"
    --exposure-column "$EXPOSURE_COLUMN"
    --hac-lags "$HAC_LAGS"
    --outdir "$ARDL_OUTDIR"
    --tag "$TAG"
    --report-name "$REPORT_NAME"
  )

  if [[ "$USE_E_Z" == "1" ]]; then
    ARDL_CMD+=(--use-e-z)
  else
    ARDL_CMD+=(--no-use-e-z)
  fi

  if [[ "$INCLUDE_WEEKDAY" == "1" ]]; then
    ARDL_CMD+=(--include-weekday)
  else
    ARDL_CMD+=(--no-include-weekday)
  fi

  if [[ "$INCLUDE_MONTH" == "1" ]]; then
    ARDL_CMD+=(--include-month)
  else
    ARDL_CMD+=(--no-include-month)
  fi

  if [[ -n "$ARDL_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_SPLIT=($ARDL_EXTRA_ARGS)
    ARDL_CMD+=("${EXTRA_SPLIT[@]}")
  fi

  if [[ -n "$ARDL_SPEC_EXTRA" ]]; then
    # shellcheck disable=SC2206
    SPEC_EXTRA_SPLIT=($ARDL_SPEC_EXTRA)
    ARDL_CMD+=("${SPEC_EXTRA_SPLIT[@]}")
  fi

  ARDL_CMD+=("${ARDL_EXTRA_FLAG[@]}")

  echo "Command: ${ARDL_CMD[*]}"
  "${ARDL_CMD[@]}"

  SUMMARY_PATH=$(ls -1 "$ARDL_OUTDIR"/"${SUMMARY_SUFFIX}"* 2>/dev/null | sort | tail -n 1)
  if [[ -z "$SUMMARY_PATH" || ! -f "$SUMMARY_PATH" ]]; then
    echo "[error] Could not locate ARDL summary for $TAG in $ARDL_OUTDIR" >&2
    exit 1
  fi

  SUMMARY_ENTRIES+=("$TAG:::${ARDL_OUTDIR}:::${SUMMARY_PATH}")

done

if [[ ${#SUMMARY_ENTRIES[@]} -gt 0 ]]; then
  mkdir -p "$(dirname "$SUMMARY_CSV")"
  step "Assemble robustness summary CSV"
  "$PYTHON_BIN" - "$SUMMARY_CSV" "${SUMMARY_ENTRIES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
entries = []
for raw in sys.argv[2:]:
    tag, outdir, summary_path = raw.split(":::", 2)
    data = json.loads(Path(summary_path).read_text())
    meta = data.get("_meta", {})
    for outcome, stats in data.items():
        if outcome == "_meta":
            continue
        row = {
            "tag": tag,
            "outcome": outcome,
            "day_panel": meta.get("day_panel"),
            "report_dir": outdir,
            "exposure_argument": meta.get("exposure_argument"),
            "exposure_column_used": meta.get("exposure_column_used"),
            "use_e_z": meta.get("use_e_z"),
            "beta_sum": stats.get("beta_sum"),
            "se_sum": stats.get("se_sum"),
            "p_sum_param": stats.get("p_sum_param"),
            "nobs": stats.get("nobs"),
            "r2": stats.get("r2"),
            "start": stats.get("start"),
            "end": stats.get("end"),
        }
        entries.append(row)

headers = [
    "tag",
    "outcome",
    "day_panel",
    "report_dir",
    "exposure_argument",
    "exposure_column_used",
    "use_e_z",
    "beta_sum",
    "se_sum",
    "p_sum_param",
    "nobs",
    "r2",
    "start",
    "end",
]

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in entries:
        writer.writerow(row)

print(f"Wrote {len(entries)} rows to {summary_csv}")
PY
fi

step "Exposure robustness pipeline complete"
echo "Summary CSV: $SUMMARY_CSV"
