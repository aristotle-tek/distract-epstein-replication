#!/usr/bin/env python3
"""Generate sanity-check diagnostics for the novelty series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running via ``python src/...`` 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if pair.empty:
        return float("nan")
    if pair["a"].std(ddof=0) == 0 or pair["b"].std(ddof=0) == 0:
        return float("nan")
    return float(pair["a"].corr(pair["b"]))


@click.command()
@click.option("--day-panel", type=click.Path(exists=True, path_type=Path), required=True,
              help="Day-level panel containing N_t, V_t, and diagnostic columns.")
@click.option("--outdir", type=click.Path(path_type=Path), default=None,
              help="Directory for diagnostic outputs. Defaults to output_root/reports/novelty_diagnostics.")
@click.option("--tag", type=str, default="", show_default=True,
              help="Optional suffix for the output folder (e.g., window names).")
def main(day_panel: Path, outdir: Path | None, tag: str) -> None:
    df = pd.read_parquet(day_panel)
    if "day" in df.columns:
        df = df.sort_values("day")

    if outdir is None:
        base = "novelty_diagnostics" if not tag else f"novelty_diagnostics_{tag}"
        outdir_path = paths.report_run_dir(base, ensure=True)
    else:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)

    N = pd.to_numeric(df.get("N_t_z", df.get("N_t")), errors="coerce")
    raw_N = pd.to_numeric(df.get("N_t"), errors="coerce")
    volume = pd.to_numeric(df.get("V_t_z", df.get("V_t")), errors="coerce")
    posts = pd.to_numeric(df.get("novelty_posts_z", df.get("n_posts")), errors="coerce")

    diagnostics: Dict[str, Any] = {
        "days_total": int(df.shape[0]),
        "novelty_missing_days": int(pd.isna(raw_N).sum()),
        "novelty_missing_share": float(pd.isna(raw_N).mean()),
        "low_sample_days": int(df.get("novelty_low_sample_flag", pd.Series(dtype=float)).fillna(0).astype(int).sum()),
        "zero_post_days": int(df.get("no_posts_flag", pd.Series(dtype=float)).fillna(0).astype(int).sum()),
        "median_posts": float(pd.to_numeric(df.get("n_posts"), errors="coerce").median(skipna=True))
        if "n_posts" in df.columns
        else float("nan"),
        "p10_posts": float(pd.to_numeric(df.get("n_posts"), errors="coerce").quantile(0.10))
        if "n_posts" in df.columns
        else float("nan"),
        "p90_posts": float(pd.to_numeric(df.get("n_posts"), errors="coerce").quantile(0.90))
        if "n_posts" in df.columns
        else float("nan"),
        "corr_novelty_volume": float(_safe_corr(N, volume)),
        "corr_novelty_posts": float(_safe_corr(N, posts)),
    }

    diagnostics_path = outdir_path / "novelty_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    histogram_path = outdir_path / "novelty_distribution.png"
    valid = raw_N.dropna()
    if not valid.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(valid, bins=30, color="#3182bd", alpha=0.85, edgecolor="white")
        plt.xlabel(r"Novelty $N_t$")
        plt.ylabel("Days")
        plt.title("Distribution of daily novelty")
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150)
        plt.close()

    top_cols = [c for c in ["day", "N_t_z", "N_t", "n_posts", "V_t"] if c in df.columns]
    if "N_t_z" in df.columns:
        top = df[top_cols].dropna(subset=["N_t_z"]).nlargest(10, "N_t_z")
    else:
        top = df[top_cols].dropna(subset=["N_t"]).nlargest(10, "N_t")
    if not top.empty:
        top_path = outdir_path / "top_novelty_days.csv"
        top.to_csv(top_path, index=False)

    click.echo(f"Wrote novelty diagnostics → {outdir_path}")


if __name__ == "__main__":
    main()
