"""Utilities for residualizing day-series outcomes before HMM analysis."""

from __future__ import annotations

import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _build_calendar_design(df: pd.DataFrame) -> pd.DataFrame:
    """Return a design matrix with calendar/baseline covariates.

    The matrix always includes an intercept. When the source  exposes
    weekday/month/inauguration indicators (as produced by
    ``src.series.day_panel``), these are expanded into dummy vars with the
    first category dropped to avoid collinearity.
    """

    cols: list[pd.DataFrame] = []

    intercept = pd.DataFrame({"intercept": 1.0}, index=df.index)
    cols.append(intercept)

    if "weekday" in df.columns:
        weekday = (
            pd.get_dummies(df["weekday"].astype(int), prefix="wd", drop_first=True)
            .astype(float)
            .reindex(df.index)
        )
        cols.append(weekday)

    if "month" in df.columns:
        month = (
            pd.get_dummies(df["month"].astype(int), prefix="m", drop_first=True)
            .astype(float)
            .reindex(df.index)
        )
        cols.append(month)

    if "inaug_2025_onward" in df.columns:
        cols.append(df[["inaug_2025_onward"]].astype(float))

    if len(cols) == 1:  # Only intercept present
        return intercept

    return pd.concat(cols, axis=1)


def _zscore(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(np.nan, index=series.index)
    mean = float(arr[mask].mean())
    std = float(arr[mask].std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index)
    z = (arr - mean) / std
    z[~mask] = np.nan
    return pd.Series(z, index=series.index)


@click.command()
@click.option("--day-panel", "day_panel_path", type=click.Path(exists=True), required=True)
@click.option("--y-col", type=str, default="N_t", show_default=True)
@click.option(
    "--out-col",
    type=str,
    default="N_t_resid_z",
    help="Column to store the residualized & z-scored series.",
    show_default=True,
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Allow overwriting an existing output parquet (defaults to False).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional destination parquet. Defaults to overwriting the input file.",
)
def cli(day_panel_path: str, y_col: str, out_col: str, overwrite: bool, out_path: Path | None) -> None:
    """Residualize an outcome against calendar covariates and z-score it."""

    in_path = Path(day_panel_path)
    df = pd.read_parquet(in_path)

    if y_col not in df.columns:
        raise click.ClickException(f"Outcome column '{y_col}' not found in {in_path}.")

    design = _build_calendar_design(df)

    y = df[y_col].astype(float)
    mask = y.notna()
    if mask.sum() < design.shape[1]:
        raise click.ClickException(
            "Not enough non-NaN observations to estimate the residualization model."
        )

    X = design.loc[mask]
    y_valid = y.loc[mask]

    model = sm.OLS(y_valid, X)
    results = model.fit()

    fitted = results.predict(X)
    resid = y_valid - fitted

    resid_full = pd.Series(np.nan, index=df.index)
    resid_full.loc[mask] = resid

    resid_z = _zscore(resid_full)

    df[out_col] = resid_z

    destination = out_path or in_path
    if destination.exists() and not overwrite and destination != in_path:
        raise click.ClickException(
            f"Output file {destination} already exists. Use --overwrite to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)

    click.echo(
        f"[residualize] Stored residualized series '{out_col}' (R^2={results.rsquared:.4f}) "
        f"at {destination}"
    )


if __name__ == "__main__":
    cli()
