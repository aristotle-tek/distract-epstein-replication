#!/usr/bin/env python3
"""stationarity diagnostics for the ARDL day panel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import click
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    column: str
    nobs: int
    adf_stat: float | None
    adf_pvalue: float | None
    adf_lags: int | None
    kpss_stat: float | None
    kpss_pvalue: float | None
    kpss_lags: int | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "column": self.column,
            "nobs": self.nobs,
            "adf_stat": self.adf_stat,
            "adf_pvalue": self.adf_pvalue,
            "adf_lags": self.adf_lags,
            "kpss_stat": self.kpss_stat,
            "kpss_pvalue": self.kpss_pvalue,
            "kpss_lags": self.kpss_lags,
        }


def _match_column(df: pd.DataFrame, name: str) -> str | None:
    lowered = name.lower()
    for col in df.columns:
        if col == name:
            return col
    for col in df.columns:
        if col.lower() == lowered:
            return col
    return None


def _resolve_exposure_column(
    df: pd.DataFrame, exposure_column: str | None, use_e_z: bool
) -> tuple[str | None, str | None]:
    base_col: str | None
    if exposure_column:
        base_col = _match_column(df, exposure_column)
        if base_col is None:
            raise click.ClickException(
                f"Exposure column '{exposure_column}' not found in the data."
            )
    else:
        base_col = _match_column(df, "E_t")

    resolved = base_col
    created = None

    if use_e_z:
        if base_col is None:
            resolved = _match_column(df, "E_t_z")
            if resolved is None:
                raise click.ClickException(
                    "Could not locate exposure column 'E_t' or 'E_t_z' in the data."
                )
            return resolved, created

        candidates: List[str] = []
        if base_col == "E_t":
            candidates.append("E_t_z")
        else:
            candidates.extend(
                [
                    f"Z_{base_col}",
                    f"z_{base_col}",
                    f"{base_col}_z",
                ]
            )
        for candidate in candidates:
            match = _match_column(df, candidate)
            if match is not None:
                resolved = match
                break
        else:
            series = pd.to_numeric(df[base_col], errors="coerce")
            mean_val = series.mean(skipna=True)
            std_val = series.std(skipna=True, ddof=0)
            if std_val and math.isfinite(std_val) and std_val > 0:
                resolved_name = "E_t_z" if base_col == "E_t" else f"Z_{base_col}"
                df[resolved_name] = (series - mean_val) / std_val
                resolved = resolved_name
                created = resolved_name
            else:
                resolved = base_col
    return resolved, created


def _prepare_series(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError("Series has fewer than 10 finite observations after cleaning.")
    return values


def _run_adf(series: np.ndarray, trend: str) -> tuple[float | None, float | None, int | None]:
    try:
        stat, pvalue, usedlag, *_ = adfuller(series, regression=trend, autolag="AIC")
    except ValueError:
        return None, None, None
    return float(stat), float(pvalue), int(usedlag)


def _run_kpss(series: np.ndarray, regression: str) -> tuple[float | None, float | None, int | None]:
    try:
        stat, pvalue, lags, *_ = kpss(series, regression=regression, nlags="auto")
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None
    return float(stat), float(pvalue), int(lags)


def run_stationarity_tests(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    adf_trend: str = "c",
    kpss_regression: str = "c",
    run_kpss_test: bool = True,
) -> list[StationarityResult]:
    results: list[StationarityResult] = []
    for name in columns:
        if name not in df.columns:
            raise KeyError(f"Column '{name}' not present in the provided DataFrame.")
        series = _prepare_series(df[name])
        adf_stat, adf_p, adf_lags = _run_adf(series, adf_trend)
        if run_kpss_test:
            kpss_stat, kpss_p, kpss_lags = _run_kpss(series, kpss_regression)
        else:
            kpss_stat = kpss_p = kpss_lags = None
        results.append(
            StationarityResult(
                column=name,
                nobs=series.size,
                adf_stat=adf_stat,
                adf_pvalue=adf_p,
                adf_lags=adf_lags,
                kpss_stat=kpss_stat,
                kpss_pvalue=kpss_p,
                kpss_lags=kpss_lags,
            )
        )
    return results


@click.command()
@click.option("--day-panel", "day_panel_path", type=click.Path(exists=True), required=True)
@click.option(
    "--outcomes",
    type=str,
    required=True,
    help="Comma-separated list of outcome columns to evaluate.",
)
@click.option(
    "--exposure-column",
    type=str,
    default=None,
    show_default=True,
    help="Exposure column to include (case-insensitive; default attempts 'E_t').",
)
@click.option("--use-e-z/--no-use-e-z", default=True, show_default=True)
@click.option(
    "--trend",
    type=click.Choice(["c", "ct", "ctt", "n"], case_sensitive=False),
    default="c",
    show_default=True,
    help="Deterministic trend assumption for the ADF test.",
)
@click.option(
    "--kpss-regression",
    type=click.Choice(["c", "ct"], case_sensitive=False),
    default="c",
    show_default=True,
    help="Deterministic component for the KPSS test.",
)
@click.option("--no-kpss", is_flag=True, help="Skip the KPSS test and run only ADF.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional csv output path for the summary table.",
)
@click.option("--additional-columns", type=str, default="", help="Extra columns to test.")
def main(
    day_panel_path: str,
    outcomes: str,
    exposure_column: str | None,
    use_e_z: bool,
    trend: str,
    kpss_regression: str,
    no_kpss: bool,
    output: Path | None,
    additional_columns: str,
) -> None:
    df = pd.read_parquet(day_panel_path)
    if "day" in df.columns:
        df = df.sort_values("day").reset_index(drop=True)

    outcome_list = [item.strip() for item in outcomes.split(",") if item.strip()]
    if not outcome_list:
        raise click.ClickException("At least one outcome column must be provided.")

    exposure_resolved, created = _resolve_exposure_column(df, exposure_column, use_e_z)

    extras = [item.strip() for item in additional_columns.split(",") if item.strip()]
    columns_to_test = list(dict.fromkeys(outcome_list + extras))
    if exposure_resolved is not None and exposure_resolved not in columns_to_test:
        columns_to_test.append(exposure_resolved)
    if created and created not in columns_to_test:
        columns_to_test.append(created)

    results = run_stationarity_tests(
        df,
        columns_to_test,
        adf_trend=trend,
        kpss_regression=kpss_regression,
        run_kpss_test=not no_kpss,
    )

    table = pd.DataFrame([res.to_dict() for res in results])
    table = table[
        [
            "column",
            "nobs",
            "adf_stat",
            "adf_pvalue",
            "adf_lags",
            "kpss_stat",
            "kpss_pvalue",
            "kpss_lags",
        ]
    ]

    click.echo("stationarity diagnostics:\n")
    click.echo(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}" if pd.notna(x) else "nan",
        )
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        click.echo(f"\nSaved summary to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
