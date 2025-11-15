#!/usr/bin/env python3
"""
Generate a styled timeline plot summarizing Fox News "Epstein" density alongside
key political milestones (from json file).

    python src/diagnostic_plots.py \
        --smooth 7 \
        --out figs/epstein_attention_timeline.pdf
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.serif": [
            "Palatino",
            "Palatino Linotype",
            "Book Antiqua",
            "URW Palladio L",
            "Nimbus Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino",
        "mathtext.it": "Palatino:italic",
        "mathtext.bf": "Palatino:bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


@dataclass
class TimelineEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    description: str | None = None
    align: str = "top"
    color: str | None = None

    @property
    def midpoint(self) -> pd.Timestamp:
        return self.start + (self.end - self.start) / 2


def _infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), here.parent, here.parent.parent, Path.cwd().parent]
    for base in candidates:
        if (base / "data_processed" / "tv_transcripts").exists():
            return base
    p = here
    for _ in range(6):
        if (p / "data_processed" / "tv_transcripts").exists():
            return p
        p = p.parent
    return Path.cwd()


def _pick_output_path(base: Path, out_path: Path | None) -> Path:
    if out_path is not None:
        return out_path.resolve()
    return (base / "out" / "figures" / "epstein_attention_timeline.pdf").resolve()


def _load_tv_series(input_path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "date" not in df.columns:
        raise click.ClickException("Input data must include a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if column not in df.columns:
        raise click.ClickException(f"Column '{column}' not found in {input_path}.")
    df = df[["date", column]].dropna(subset=[column]).sort_values("date")
    if df.empty:
        raise click.ClickException("No rows available after parsing the TV data.")
    return df.rename(columns={column: "value"})


def _load_events(events_path: Path | None) -> list[TimelineEvent]:
    if events_path is None:
        return []
    events_path = events_path.resolve()
    if not events_path.exists():
        click.secho(f"Event file not found: {events_path}; continuing without annotations.", fg="yellow")
        return []
    with events_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    events: list[TimelineEvent] = []
    for row in payload:
        start = pd.to_datetime(row["start"]).normalize()
        end = pd.to_datetime(row.get("end") or row["start"]).normalize()
        label = row.get("label")
        if not label:
            raise click.ClickException("Each event must contain a 'label'.")
        align = row.get("align", "top").lower()
        color = row.get("color")
        events.append(
            TimelineEvent(
                start=start,
                end=end,
                label=label,
                description=row.get("description"),
                align=align if align in {"top", "bottom"} else "top",
                color=color,
            )
        )
    return events


def _load_google_series(
    google_path: Path,
    date_col: str,
    value_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df = pd.read_csv(google_path)
    if date_col not in df or value_col not in df:
        raise click.ClickException(f"Google Trends file must include '{date_col}' and '{value_col}'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (
        df.rename(columns={date_col: "date", value_col: "value"})
        .dropna(subset=["date", "value"])
        .sort_values("date")
    )
    mask = (df["date"] >= start) & (df["date"] <= end)
    df = df.loc[mask]
    if df.empty:
        click.secho("Google Trends data is empty in the requested window; skipping overlay.", fg="yellow")
    return df


def _add_event_annotations(ax: plt.Axes, events: Iterable[TimelineEvent]) -> None:
    events = list(events)
    if not events:
        return
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for event in events:
        color = event.color or "#8c2d04"
        if event.end > event.start:
            ax.axvspan(event.start, event.end, color=color, alpha=0.08, linewidth=0)
            x_pos = event.midpoint
        else:
            x_pos = event.start
        ax.axvline(x_pos, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
        y = 1.03
        va = "bottom"
        ax.text(
            x_pos,
            y,
            event.label,
            rotation=90,
            ha="center",
            va=va,
            fontsize=9,
            color=color,
            transform=transform,
            fontweight="bold",
        )


def _prepare_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    window = df.loc[mask].copy()
    if window.empty:
        raise click.ClickException("No observations fall inside the requested window.")
    return window


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to merged TV CSV (defaults to repo-root/data_processed/tv_transcripts/daily_mg_epstein.csv).",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output PDF/PNG path for the timeline.",
)
@click.option(
    "-c",
    "--column",
    "series_column",
    default="daily_fox_epstein_density",
    show_default=True,
    help="Column to plot from the TV CSV.",
)
@click.option(
    "--events",
    "events_path",
    type=click.Path(path_type=Path),
    default=None,
    help="JSON file with timeline annotations.",
)
@click.option(
    "--start-date",
    default="2024-10-08",
    show_default=True,
    help="Left bound of the plot window (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    default="2025-10-17",
    show_default=True,
    help="Right bound of the plot window (YYYY-MM-DD).",
)
@click.option(
    "-s",
    "--smooth",
    "smooth_n",
    type=click.IntRange(1, None),
    default=7,
    show_default=True,
    help="Rolling window (days) applied to highlight the signal.",
)
@click.option(
    "--google-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional Google Trends CSV to overlay (scaled to the TV series).",
)
@click.option(
    "--google-date-col",
    default="date",
    show_default=True,
    help="Date column name within the Google Trends file.",
)
@click.option(
    "--google-value-col",
    default="value",
    show_default=True,
    help="Value column name within the Google Trends file.",
)
@click.option(
    "--google-scale",
    type=float,
    default=None,
    help="Manual multiplier for the Google series (defaults to matching the TV max).",
)
def main(
    input_path: Path | None,
    out_path: Path | None,
    series_column: str,
    events_path: Path | None,
    start_date: str,
    end_date: str,
    smooth_n: int,
    google_path: Path | None,
    google_date_col: str,
    google_value_col: str,
    google_scale: float | None,
) -> None:
    base = _infer_repo_root()
    if input_path is None:
        input_path = (base / "data_processed" / "tv_transcripts" / "daily_mg_epstein.csv").resolve()
    if events_path is None:
        events_path = (base / "data_processed" / "timelines" / "epstein_attention_events.json").resolve()
    out_path = _pick_output_path(base, out_path)

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if end_ts <= start_ts:
        raise click.ClickException("end-date must come after start-date.")

    tv_df = _load_tv_series(input_path, series_column)
    window_df = _prepare_window(tv_df, start_ts, end_ts)

    raw = window_df["value"]
    smooth = raw.rolling(window=smooth_n, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    ax.plot(window_df["date"], raw, color="#b5c7de", linewidth=1.0, alpha=0.85, label="Daily density")
    ax.plot(window_df["date"], smooth, color="#08306b", linewidth=2.4, label=f"{smooth_n}-day mean")

    google_df = None
    if google_path is not None:
        google_df = _load_google_series(google_path, google_date_col, google_value_col, start_ts, end_ts)
        if not google_df.empty:
            scale = google_scale
            if scale is None:
                max_tv = smooth.max()
                max_google = google_df["value"].max()
                scale = (max_tv / max_google) if max_google else 1.0
            google_df["value_scaled"] = google_df["value"] * scale
            ax.plot(
                google_df["date"],
                google_df["value_scaled"],
                color="#d95f0e",
                linestyle=(0, (4, 2)),
                linewidth=1.2,
                alpha=0.7,
                label="Google Trends (scaled)",
            )

    ax.set_title("") #Fox News 'Epstein' attention with key milestones")
    ax.set_ylabel("Density per 1,000 words (Fox News)")
    ax.set_xlabel("Date")
    ax.set_xlim(start_ts, end_ts)

    y_max_candidates = [raw.max(), smooth.max()]
    if google_df is not None and not google_df.empty:
        y_max_candidates.append(google_df["value_scaled"].max())
    valid = [v for v in y_max_candidates if pd.notnull(v)]
    y_max = (max(valid) * 1.15) if valid else 1.0
    ax.set_ylim(0, max(y_max, 0.1))

    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    ax.grid(axis="y", color="#cfd8e3", linewidth=0.8, linestyle="--", alpha=0.6)

    events = [
        event
        for event in _load_events(events_path)
        if (event.end >= start_ts) and (event.start <= end_ts)
    ]
    _add_event_annotations(ax, events)

    ax.legend(loc="upper left", frameon=False, handlelength=3)

    fig.subplots_adjust(top=0.78, bottom=0.28, left=0.1, right=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    click.echo(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
