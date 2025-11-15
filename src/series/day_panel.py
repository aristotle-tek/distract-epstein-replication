"""
Sets up the data for e.g. ardl.py

-- example with fox_expstein_density
python src/series/day_panel.py build \
  --novelty-parquet   data_processed/novelty_energy.parquet \
  --epstein-file      data_processed/tv_transcripts/daily_mg_epstein.csv \
  --epstein-column daily_fox_epstein_density \
  --start 2024-10-01 --end 2025-10-19 \
  --tag tv_epstein_daily_fox_density \
  --out data_processed/runs/{tag}/day_panel.parquet



"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import click
import numpy as np
import pandas as pd

# Allow running via `python src/...`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths, run_metadata


import os


def _read_novelty(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    cols = ["day"]
    for name in ["N_t", "n_posts", "n_ref_posts", "min_posts_used", "min_ref_posts_used"]:
        if name in df.columns:
            cols.append(name)
    return df[cols].sort_values("day")



def _read_epstein(
    path: Path,
    series_name: str | None = None,
    value_column: str | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    day_key = next((k for k in ("day", "date") if k in cols), None)
    if day_key is None:
        raise ValueError("Epstein csv/ parquet must include a 'day'/'date' column.")
    day_col = cols[day_key]

    candidate_lookup = {c.lower(): c for c in df.columns if c.lower() != day_key}

    if series_name is not None and "series" in cols:
        s_col = cols["series"]
        df = df[df[s_col].astype(str) == series_name]

    numeric_cols: list[str] = []
    for raw_col in list(candidate_lookup.values()):
        if raw_col == day_col:
            continue
        converted = pd.to_numeric(df[raw_col], errors="coerce")
        if converted.notna().any():
            df[raw_col] = converted
            numeric_cols.append(raw_col)

    if not numeric_cols:
        raise ValueError("No numeric columns avail for Epstein measure.")

    chosen_col: str | None = None

    if value_column is not None:
        # match case-insensitively against avail numeric columns
        for raw_col in numeric_cols:
            if raw_col.lower() == value_column.lower():
                chosen_col = raw_col
                break
        if chosen_col is None:
            raise ValueError(
                f"Requested epstein column '{value_column}' not found. Available columns: {sorted(numeric_cols)}"
            )
    else:
        preferred_order = [
            "daily_fox_epstein_density",
            "daily_all3_epstein_mean",
            "daily_fox_epstein_hits",
            "daily_all3_epstein_hits",
        ]
        for candidate in preferred_order:
            if candidate in numeric_cols:
                chosen_col = candidate
                break
        if chosen_col is None:
            # fall back to first numeric column for backwards compatibility
            chosen_col = numeric_cols[0]

    df = df[[day_col] + numeric_cols].copy()
    df.rename(columns={day_col: "day"}, inplace=True)
    df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce").dt.floor("D")

    duplicate_mask = df.duplicated(subset=["day"], keep=False)
    if duplicate_mask.any():
        duplicate_days = df.loc[duplicate_mask, "day"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(
            "Epstein input contains multiple rows for the same day. "
            "Aggregation would distort exposure magnitudes. "
            f"Duplicate days: {sorted(duplicate_days)}"
        )

    df.sort_values("day", inplace=True)

    df["E_t"] = df[chosen_col]
    return df, chosen_col, numeric_cols



def _read_posts_volume(posts_parquet: Path) -> pd.DataFrame:
    # id, created_at, content, content_clean, day, ...
    use = ["id", "day"]
    df = pd.read_parquet(posts_parquet, columns=use)
    df["day"] = pd.to_datetime(df["day"], utc=True).dt.floor("D")
    vol = df.groupby("day")["id"].count().rename("V_t").reset_index()
    return vol.sort_values("day")


def _read_bucket_series(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    return df.sort_values("day")


def _zscore(s: pd.Series) -> pd.Series:
    m = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - m) / sd


def _calendar_union(frames: list[pd.DataFrame], start: str | None, end: str | None) -> pd.DataFrame:
    mins, maxs = [], []
    for df in frames:
        if "day" in df.columns and not df.empty:
            mins.append(df["day"].min())
            maxs.append(df["day"].max())
    if not mins:
        raise ValueError("No input frames contained a 'day' column.")
    lo = pd.to_datetime(start, utc=True) if start else min(mins)
    hi = pd.to_datetime(end, utc=True) if end else max(maxs)
    cal = pd.DataFrame({"day": pd.date_range(lo, hi, freq="D", tz="UTC")})
    return cal


def _add_calendar_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["weekday"] = p["day"].dt.weekday
    p["month"] = p["day"].dt.month
    p["inaug_2025_onward"] = (p["day"] >= pd.Timestamp("2025-01-20", tz="UTC")).astype(int)
    return p


def _auto_components(
    tag: str | None,
    epstein_file: Path,
    start: str | None,
    end: str | None,
    posts_parquet: Path | None,
    epstein_series_name: str | None,
    epstein_column: str | None,
) -> tuple[str, ...]:
    if tag:
        parts = [p for p in Path(tag).parts if p not in {".", "..", ""}]
    else:
        parts = [epstein_file.stem]
        if epstein_series_name:
            parts.append(epstein_series_name)
        if epstein_column:
            parts.append(epstein_column)
        if start or end:
            lo = start or "start"
            hi = end or "end"
            parts.append(f"{lo}_to_{hi}")
        if posts_parquet:
            parts.append("with_volume")
    return ("day_panel", *parts)


def _resolve_out_path(out: Path | None, run_dir: Path, tag_slug: str) -> Path:
    if out is None:
        return run_dir / "day_panel.parquet"
    s = str(out)
    if "{tag}" in s:
        return Path(s.format(tag=tag_slug))
    p = Path(out)
    if p.suffix == "":  # likely a directory
        return p / "day_panel.parquet"
    return p


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None,
              help="Optional path to configuration file overriding config.yml")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None):
    if config_path:
        paths.configure(config_path)
    ctx.obj = {"config_path": config_path}


@cli.command("build")
@click.option(
    "--novelty-parquet",
    type=click.Path(path_type=Path),
    default=None,
    help="Novelty parquet (default: paths.novelty_energy_parquet())",
)
@click.option(
    "--epstein-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Parquet or CSV with columns: day/date + at least one numeric Epstein measure.",
)
@click.option(
    "--bucket-parquet",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional parquet with daily bucket shares (e.g., from src.llm_pipeline.truth_social_llm_buckets).",
)
@click.option(
    "--epstein-column",
    type=str,
    required=False,
    default=None,
    help="Column to use for the primary E_t series (default auto-detect).",
)
@click.option(
    "--posts-parquet",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
    help="Optional: posts parquet for volume (omit to skip computing V_t).",
)
@click.option(
    "--epstein-series-name",
    type=str,
    required=False,
    default=None,
    help="If CSV has a 'Series' column, filter to this name (e.g., 'Volume Intensity').",
)
@click.option("--start", type=str, required=False, default=None, help="YYYY-MM-DD (UTC)")
@click.option("--end", type=str, required=False, default=None, help="YYYY-MM-DD (UTC)")
@click.option("--out", type=click.Path(path_type=Path), required=False, default=None,
              help="Output path. Defaults to paths.runs_dir()/{tag}/day_panel.parquet. "
                   "If it contains '{tag}', that placeholder is expanded. "
                   "If it points to a directory, the file name 'day_panel.parquet' is used.")
@click.option("--tag", type=str, required=False, default=None,
              help="Run tag used for runs/{tag}. If omitted, a short fingerprint is used.")
def build_day_panel(
    novelty_parquet: Path | None,
    epstein_file: Path,
    epstein_column: str | None,
    bucket_parquet: Path | None,
    posts_parquet: Path | None,
    epstein_series_name: str | None,
    start: str | None,
    end: str | None,
    out: Path | None,
    tag: str | None,
):
    novelty_parquet = Path(novelty_parquet) if novelty_parquet else paths.novelty_energy_parquet()
    posts_parquet = Path(posts_parquet) if posts_parquet else None
    nov = _read_novelty(novelty_parquet)
    epi, chosen_epstein_col, available_epstein_cols = _read_epstein(
        epstein_file, epstein_series_name, epstein_column
    )
    bucket = None
    if bucket_parquet:
        bucket_parquet = Path(bucket_parquet)
        bucket = _read_bucket_series(bucket_parquet)
    frames = [nov, epi]
    if bucket is not None:
        frames.append(bucket)
    if posts_parquet:
        vol = _read_posts_volume(posts_parquet)
        frames.append(vol)

    cal = _calendar_union(frames, start, end)

    panel = cal.merge(nov, on="day", how="left") \
               .merge(epi, on="day", how="left")

    if posts_parquet:
        panel = panel.merge(vol, on="day", how="left")
        panel["no_posts_flag"] = panel["V_t"].fillna(0).eq(0).astype(int)
        panel["V_t_z"] = _zscore(panel["V_t"])

    panel = _add_calendar_covariates(panel)
    panel["E_t_z"] = _zscore(panel["E_t"])

    epi_value_cols = [c for c in available_epstein_cols if c in panel.columns]
    for col in epi_value_cols:
        panel[f"Z_{col}"] = _zscore(panel[col])
    if "N_t" in panel:
        panel["N_t_z"] = _zscore(panel["N_t"])
        panel["novelty_missing_flag"] = panel["N_t"].isna().astype(int)

    if "n_posts" in panel.columns:
        panel["novelty_posts_z"] = _zscore(panel["n_posts"])
    if "n_ref_posts" in panel.columns:
        panel["novelty_ref_posts_z"] = _zscore(panel["n_ref_posts"])
    if {"n_posts", "min_posts_used"}.issubset(panel.columns):
        default_thresh = panel["min_posts_used"].dropna().max()
        if pd.isna(default_thresh):
            default_thresh = 0.0
        threshold = panel["min_posts_used"].fillna(default_thresh)
        panel["novelty_low_sample_flag"] = panel["n_posts"].fillna(0).lt(threshold).astype(int)


    bucket_share_cols = [c for c in panel.columns if c.startswith("bucket_") and c.endswith("_share")]
    for c in bucket_share_cols:
        panel[f"Z_{c}"] = _zscore(panel[c])
    if "media_only_share" in panel.columns:
        panel["Z_media_only_share"] = _zscore(panel["media_only_share"])

    #  runs/ {tag} layout
    components = _auto_components(
        tag,
        epstein_file,
        start,
        end,
        posts_parquet,
        epstein_series_name,
        chosen_epstein_col,
    )
    run_dir = run_metadata.run_directory(*components, ensure=True)
    clean_components = run_dir.relative_to(paths.runs_dir()).parts
    tag_slug = "/".join(clean_components)

    out_path = _resolve_out_path(out, run_dir, tag_slug)
    paths.ensure_directory(out_path.parent)
    panel.to_parquet(out_path, index=False)

    # (symlink)
    latest_link: Path | None = None
    try:
        link = paths.day_panel_parquet()
        paths.ensure_directory(link.parent)
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.relpath(out_path, link.parent), link)
        latest_link = link
    except OSError:
        pass

    meta = run_metadata.build_metadata(
        components=clean_components,
        parameters={
            "start": start,
            "end": end,
            "epstein_series_name": epstein_series_name,
            "epstein_column": chosen_epstein_col,
            "tag_argument": tag,
        },
        inputs={
            "novelty_parquet": novelty_parquet,
            "epstein_file": epstein_file,
            "bucket_parquet": bucket_parquet,
            "posts_parquet": posts_parquet,
        },
        outputs={
            "day_panel": out_path,
            "latest_symlink": latest_link,
        },
        extras={
            "n_days": int(panel.shape[0]),
            "start_day": panel["day"].min(),
            "end_day": panel["day"].max(),
            "columns": list(panel.columns),
            "epstein_available_columns": available_epstein_cols,
            "bucket_share_columns": bucket_share_cols,
        },
    )

    run_metadata.write_metadata(run_dir, meta)
    print(json.dumps(meta, indent=2, default=str))
    print(f"Saved day panel → {out_path}")


if __name__ == "__main__":
    cli()
