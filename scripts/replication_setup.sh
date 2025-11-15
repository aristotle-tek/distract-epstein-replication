#!/usr/bin/env bash
# Prepare data for analyses: ingest, embed, aggregate transcripts,
# and build the day-level analysis panel. 

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

INSTALL_DEPS=${INSTALL_DEPS:-0}
WINDOW_START=${WINDOW_START:-2024-10-01}
WINDOW_END=${WINDOW_END:-2025-10-19}
RUN_TV_AGGREGATION=${RUN_TV_AGGREGATION:-0}

RAW_TRUTH_CSV=${RAW_TRUTH_CSV:-$ROOT/data_raw/truth_social/truth_archive.csv}
TRUTH_PARQUET=${TRUTH_PARQUET:-$ROOT/data_raw/truth_social/truth_posts.parquet}
EMB_PARQUET=${EMB_PARQUET:-$ROOT/data_processed/post_embeddings.parquet}
NOVELTY_PARQUET=${NOVELTY_PARQUET:-$ROOT/data_processed/novelty_energy.parquet}
BUCKET_PARQUET=${BUCKET_PARQUET:-$ROOT/data_processed/truth_buckets/daily_shares.parquet}
TV_DIR=${TV_DIR:-$ROOT/data_processed/tv_transcripts}
FOX_CACHE_DIR=${FOX_CACHE_DIR:-$ROOT/data_transcripts}
CNN_CACHE_DIR=${CNN_CACHE_DIR:-$ROOT/data_transcripts_cnn}
MSNBC_CACHE_DIR=${MSNBC_CACHE_DIR:-$ROOT/data_transcripts_msnbc}
FOX_DAILY=${FOX_DAILY:-$TV_DIR/daily_fox_epstein.csv}
CNN_DAILY=${CNN_DAILY:-$TV_DIR/daily_cnn_epstein.csv}
MSNBC_DAILY=${MSNBC_DAILY:-$TV_DIR/daily_msnbc_epstein.csv}
COMBINED_DAILY=${COMBINED_DAILY:-$TV_DIR/daily_mg_epstein.csv}
EPSTEIN_COLUMN=${EPSTEIN_COLUMN:-daily_fox_epstein_density}
DAY_PANEL_TAG=${DAY_PANEL_TAG:-tv_epstein_daily_fox_density}
DAY_PANEL=${DAY_PANEL:-$ROOT/data_processed/runs/day_panel/$DAY_PANEL_TAG/day_panel.parquet}

export TV_DIR FOX_DAILY CNN_DAILY MSNBC_DAILY COMBINED_DAILY

if [[ ! -f "$RAW_TRUTH_CSV" ]]; then
  echo "[error] Missing Truth Social CSV: $RAW_TRUTH_CSV" >&2
  exit 1
fi


step() {
  echo >&2 "\n===================="
  echo >&2 "[step] $1"
  echo >&2 "====================\n"
}

if [[ "$INSTALL_DEPS" != "0" ]]; then
  step "Install Python dependencies"
  $PYTHON_BIN -m pip install --upgrade pip
  $PYTHON_BIN -m pip install -r requirements.txt
fi

step "Convert Truth Social CSV to Parquet"
$PYTHON_BIN src/data_prep/ingest_csv.py --config "$CONFIG_PATH" \
  --csv "$RAW_TRUTH_CSV" \
  --out "$TRUTH_PARQUET"


step "Compute daily novelty (energy distance)"
$PYTHON_BIN src/novelty.py --config "$CONFIG_PATH" build \
  --method energy \
  --emb-path "$EMB_PARQUET" \
  --window 7 \
  --start "$WINDOW_START" \
  --end "$WINDOW_END" \
  --out "$NOVELTY_PARQUET"

if [[ "$RUN_TV_AGGREGATION" != "0" ]]; then
  mkdir -p "$TV_DIR"

  for cache in "$FOX_CACHE_DIR" "$CNN_CACHE_DIR" "$MSNBC_CACHE_DIR"; do
    if [[ ! -d "$cache" ]]; then
      echo "[error] Missing transcript cache directory: $cache" >&2
      echo "Fetch transcripts with src/ia_tv_get.py before running this script." >&2
      exit 1
    fi
  done

  step "Aggregate Fox News transcripts"
  $PYTHON_BIN src/ia_tv_analyze.py \
    --cache-dir "$FOX_CACHE_DIR" \
    --start "$WINDOW_START" \
    --end "$WINDOW_END" \
    --term "Epstein" \
    --clean auto \
    --per-show-out "$TV_DIR/per_show_fox_epstein.csv" \
    --daily-out "$FOX_DAILY" \
    -v

  step "Aggregate CNN transcripts"
  $PYTHON_BIN src/ia_tv_analyze.py \
    --cache-dir "$CNN_CACHE_DIR" \
    --start "$WINDOW_START" \
    --end "$WINDOW_END" \
    --term "Epstein" \
    --clean auto \
    --per-show-out "$TV_DIR/per_show_cnn_epstein.csv" \
    --daily-out "$CNN_DAILY" \
    -v

  step "Aggregate MSNBC transcripts"
  $PYTHON_BIN src/ia_tv_analyze.py \
    --cache-dir "$MSNBC_CACHE_DIR" \
    --start "$WINDOW_START" \
    --end "$WINDOW_END" \
    --term "Epstein" \
    --clean auto \
    --per-show-out "$TV_DIR/per_show_msnbc_epstein.csv" \
    --daily-out "$MSNBC_DAILY" \
    -v

  step "Combine network-level daily Epstein series"
  $PYTHON_BIN src/data_loaders.py

  if [[ "$COMBINED_DAILY" != "$TV_DIR/daily_mg_epstein.csv" ]]; then
    if [[ -f "$TV_DIR/daily_mg_epstein.csv" ]]; then
      mkdir -p "$(dirname "$COMBINED_DAILY")"
      cp "$TV_DIR/daily_mg_epstein.csv" "$COMBINED_DAILY"
    fi
  fi
else
  echo "[info] Skipping TV transcript aggregation (set RUN_TV_AGGREGATION=1 to enable)." >&2
fi

if [[ ! -f "$COMBINED_DAILY" ]]; then
  echo "[error] Combined daily Epstein series not found: $COMBINED_DAILY" >&2
  echo "Provide a sample file via COMBINED_DAILY or rerun with RUN_TV_AGGREGATION=1." >&2
  exit 1
fi

if [[ -n "$BUCKET_PARQUET" && ! -f "$BUCKET_PARQUET" ]]; then
  echo "[warn] Bucket share parquet not found at: $BUCKET_PARQUET" >&2
  echo "       The day panel will be built without bucket covariates." >&2
fi

step "Build day-level analysis panel"
DAY_PANEL_ARGS=(
  --novelty-parquet "$NOVELTY_PARQUET"
  --epstein-file "$COMBINED_DAILY"
  --epstein-column "$EPSTEIN_COLUMN"
  --posts-parquet "$TRUTH_PARQUET"
  --start "$WINDOW_START"
  --end "$WINDOW_END"
  --tag "$DAY_PANEL_TAG"
)

if [[ -n "$BUCKET_PARQUET" && -f "$BUCKET_PARQUET" ]]; then
  DAY_PANEL_ARGS+=(--bucket-parquet "$BUCKET_PARQUET")
fi

$PYTHON_BIN src/series/day_panel.py --config "$CONFIG_PATH" build "${DAY_PANEL_ARGS[@]}"

step "Replication data preparation complete"
echo "Day panel written to: $DAY_PANEL"
