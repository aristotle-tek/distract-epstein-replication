#!/usr/bin/env python3
"""Plot daily Epstein exposure alongside Truth Social bucket shares.

This script visualises the day-to-day movement of the Fox News Epstein
exposure series together with the Truth Social bucket shares exported by the
``src/llm_pipeline/truth_social_llm_buckets.py`` workflow.  By default the figure uses a
large "full page" size to make short-term movements easier to see, but a
compact size is available via a flag.

Example usage::

python -m src.llm_pipeline.truth_bucket_daily_plot \
    --bucket-parquet data_processed/truth_buckets/daily_shares.parquet \
    --epstein-path data_processed/tv_transcripts/daily_mg_epstein.csv \
    --epstein-column daily_fox_epstein_density \
    --smooth 7 --start-date 2025-01-01 --end-date 2025-10-21 \
    --drop-bucket other \
    --drop-bucket elections_fraud \
    --drop-bucket media_combat

"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Allow running with ``python -m src...``
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_pipeline.truth_social_llm_buckets import BUCKET_DEFINITIONS
from src.utils import paths


CONFIG_ENV_VAR = "TRCIRCUS_CONFIG"
DEFAULT_EPSTEIN_PATH = Path("data_processed/tv_transcripts/daily_mg_epstein.csv")


def _resolve_config(config_path: Path | None) -> None:
    """config the shared path helpers, respecting the environment override."""

    if config_path is None:
        env_cfg = os.environ.get(CONFIG_ENV_VAR)
        if env_cfg:
            config_path = Path(env_cfg)

    if config_path is not None:
        paths.configure(config_path)


def _load_table(table_path: Path) -> pd.DataFrame:
    """Load a csv or parquet file to df"""

    if not table_path.exists():
        raise click.ClickException(f"Input not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(table_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(table_path)
    else:
        raise click.ClickException(
            f"Unsupported file extension for {table_path}. Use CSV or Parquet."
        )
    return df


_ALIAS_SANITISE_RE = re.compile(r"[^a-z0-9]+")


DISPLAY_LABEL_OVERRIDES = {
    "courts_trials": "Law and order; Federal troop deployments",
    "economy_inflation": "Economy; Inflation; Trade",
}


SERIES_LABEL_OVERRIDES = {
    "daily_fox_epstein_density": "Density of 'Epstein' in Fox News TV transcripts",
}


def _display_label(bucket: str) -> str:
    """Return the legend-friendly label for a bucket slug."""

    if bucket in DISPLAY_LABEL_OVERRIDES:
        return DISPLAY_LABEL_OVERRIDES[bucket]
    return bucket.replace("_", " ").title()


def _metric_label(series_name: str) -> str:
    """Return a readable label for auxiliary series such as Epstein density."""

    if series_name in SERIES_LABEL_OVERRIDES:
        return SERIES_LABEL_OVERRIDES[series_name]
    return series_name.replace("_", " ")


def _sanitise_alias(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters for name matching."""

    return _ALIAS_SANITISE_RE.sub("", value.lower())


def _normalise_drop_buckets(raw_names: Iterable[str]) -> set[str]:
    """Translate user-provided bucket names into canonical bucket slugs."""

    aliases: dict[str, str] = {}
    for bucket in BUCKET_DEFINITIONS:
        pretty = bucket.replace("_", " ")
        display = _display_label(bucket)
        segments = [segment.strip() for segment in display.split(";") if segment.strip()]
        candidates = {
            bucket,
            pretty,
            pretty.title(),
            pretty.replace(" ", ""),
            display,
            display.replace(";", " "),
            display.replace(" ", ""),
        }
        for segment in segments:
            candidates.add(segment)
            candidates.add(segment.replace(" ", ""))
        for variant in candidates:
            key = _sanitise_alias(variant)
            if key:
                aliases[key] = bucket

    unknown: list[str] = []
    resolved: set[str] = set()
    for name in raw_names:
        key = _sanitise_alias(name)
        if key in aliases:
            resolved.add(aliases[key])
        else:
            unknown.append(name)

    if unknown:
        available = ", ".join(sorted(_display_label(label) for label in BUCKET_DEFINITIONS))
        raise click.ClickException(
            "Unknown bucket name(s) to drop: "
            + ", ".join(unknown)
            + f". Available buckets: {available}."
        )

    return resolved


def _prepare_bucket_frame(
    table_path: Path,
    drop_buckets: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return the bucket df and the ordered share columns."""

    df = _load_table(table_path)
    if "day" not in df.columns:
        raise click.ClickException(
            "Bucket parquet is missing a 'day' column. Did you pass the daily shares file?"
        )

    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
    if df["day"].isna().any():
        raise click.ClickException("Found unparsable dates in bucket data.")

    bucket_order = list(BUCKET_DEFINITIONS.keys())
    share_columns: list[str] = []
    for bucket in bucket_order:
        if bucket in drop_buckets:
            continue
        col = f"bucket_{bucket}_share"
        if col in df.columns:
            share_columns.append(col)

    if not share_columns:
        share_candidates = [c for c in df.columns if c.startswith("bucket_") and c.endswith("_share")]
        if not share_candidates:
            raise click.ClickException("No bucket share columns found in the provided file.")
        share_columns = [
            col
            for col in sorted(share_candidates)
            if col.removeprefix("bucket_").removesuffix("_share") not in drop_buckets
        ]
    if not share_columns:
        drop_pretty = ", ".join(sorted(_display_label(name) for name in drop_buckets))
        raise click.ClickException(
            "No bucket share columns remain after applying exclusions"
            + (f" ({drop_pretty})." if drop_pretty else ".")
        )

    # Convert to % for friendlier scaling
    df[share_columns] = df[share_columns] * 100.0
    return df[["day", *share_columns]].sort_values("day"), share_columns


def _prepare_epstein_series(
    table_path: Path,
    column: str,
) -> pd.DataFrame:
    df = _load_table(table_path)
    if "date" in df.columns:
        df = df.rename(columns={"date": "day"})
    if "day" not in df.columns:
        raise click.ClickException("Exposure file must contain a 'date' or 'day' column.")

    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
    if df["day"].isna().any():
        raise click.ClickException("Found unparsable dates in exposure data.")

    if column not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise click.ClickException(
            f"Column '{column}' not present in exposure data. Available columns: {available}"
        )

    df = df[["day", column]].rename(columns={column: "epstein"})
    return df.sort_values("day")


def _restrict_date_range(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if start:
        start_ts = pd.Timestamp(start).tz_localize("UTC")
        df = df[df["day"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end).tz_localize("UTC")
        df = df[df["day"] <= end_ts]
    return df


def _apply_smoothing(df: pd.DataFrame, columns: Sequence[str], window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    smoothed = df.copy()
    smoothed[list(columns)] = (
        df[list(columns)]
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )
    return smoothed


def _format_bucket_label(column: str) -> str:
    bucket = column.removeprefix("bucket_").removesuffix("_share")
    return _display_label(bucket)


def _choose_figsize(full_page: bool) -> tuple[float, float]:
    return (17.5, 10.5) if full_page else (12.0, 6.5)


def _configure_ticks(ax: plt.Axes, tick_days: Iterable[int]) -> None:
    tick_days = list(dict.fromkeys(sorted(int(d) for d in tick_days if 1 <= int(d) <= 31)))
    if not tick_days:
        tick_days = [1, 15]
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=tick_days))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=45)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Optional path to config.yml (defaults to $TRCIRCUS_CONFIG or repo config).",
)
@click.option(
    "--bucket-parquet",
    type=click.Path(path_type=Path),
    default=None,
    help="Daily Truth Social bucket parquet (defaults to config paths.truth_buckets/daily_shares.parquet).",
)
@click.option(
    "--epstein-path",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV or Parquet containing the Epstein exposure series.",
)
@click.option(
    "--epstein-column",
    type=str,
    default="daily_fox_epstein_density",
    show_default=True,
    help="Column in the exposure table to plot.",
)
@click.option(
    "--start-date",
    type=str,
    default=None,
    help="Filter start date (inclusive, YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="Filter end date (inclusive, YYYY-MM-DD).",
)
@click.option(
    "--smooth",
    "smooth_window",
    type=click.IntRange(0, None),
    default=0,
    show_default=True,
    help="Rolling window (days) for smoothing. Use 0 or 1 for no smoothing.",
)
@click.option(
    "--drop-other",
    is_flag=True,
    help="Exclude the 'other' bucket from the plot.",
)
@click.option(
    "--drop-bucket",
    "drop_buckets",
    type=str,
    multiple=True,
    help="Name of a bucket to exclude (repeat flag to drop multiple; case-insensitive).",
)
@click.option(
    "--standard-size/--full-page",
    "full_page",
    default=True,
    help="Switch between standard and full-page figure sizes (default full-page).",
)
@click.option(
    "--tick-day",
    "tick_days",
    type=int,
    multiple=True,
    help="Specific days of the month to label on the x-axis (repeat flag for multiples).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output path. Defaults to figures/truth_bucket_epstein_daily.png.",
)
def main(
    config_path: Path | None,
    bucket_parquet: Path | None,
    epstein_path: Path | None,
    epstein_column: str,
    start_date: str | None,
    end_date: str | None,
    smooth_window: int,
    drop_other: bool,
    drop_buckets: Sequence[str],
    full_page: bool,
    tick_days: Sequence[int],
    out_path: Path | None,
) -> None:

    _resolve_config(config_path)

    if bucket_parquet is None:
        bucket_parquet = paths.truth_bucket_daily_shares_parquet()
    if epstein_path is None:
        epstein_path = paths.repo_path(DEFAULT_EPSTEIN_PATH)
        if not epstein_path.exists():
            epstein_path = DEFAULT_EPSTEIN_PATH

    excluded = _normalise_drop_buckets(drop_buckets)
    if drop_other:
        excluded.add("other")

    bucket_df, share_columns = _prepare_bucket_frame(Path(bucket_parquet), excluded)
    epstein_df = _prepare_epstein_series(Path(epstein_path), epstein_column)

    merged = pd.merge(bucket_df, epstein_df, on="day", how="inner")
    merged = _restrict_date_range(merged, start_date, end_date)
    if merged.empty:
        raise click.ClickException("No data available after applying filters.")

    merged = _apply_smoothing(merged, [*share_columns, "epstein"], smooth_window)

    # Drop timezone info for matplotlib (displays better)
    merged["day"] = merged["day"].dt.tz_convert("UTC").dt.tz_localize(None)

    fig, ax_left = plt.subplots(figsize=_choose_figsize(full_page))

    for column in share_columns:
        ax_left.plot(merged["day"], merged[column], label=_format_bucket_label(column))

    ax_left.set_ylabel("% of Trump Truth Social posts on topic")
    ax_left.set_xlabel("Date")

    right_label = _metric_label(epstein_column)
    ax_right = ax_left.twinx()
    ax_right.plot(merged["day"], merged["epstein"], color="black", linewidth=2, label=right_label)
    ax_right.set_ylabel(right_label)

    _configure_ticks(ax_left, tick_days if tick_days else (1, 15))

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")

    title = ""
    ax_left.set_title(title)

    fig.tight_layout()

    if out_path is None:
        out_path = paths.figures_dir(ensure=True) / "truth_bucket_epstein_daily.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200)
    click.echo(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
