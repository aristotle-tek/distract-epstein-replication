import sys
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze import stationarity


def _make_stationary_series(n: int = 400, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=1.0, size=n)
    series = np.zeros(n)
    phi = 0.6
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    return series


def _make_random_walk(n: int = 400, seed: int = 456) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=1.0, size=n))


def test_stationarity_tests_reject_unit_root():
    df = pd.DataFrame({"Y": _make_stationary_series()})
    results = stationarity.run_stationarity_tests(df, ["Y"])
    assert len(results) == 1
    res = results[0]
    assert res.adf_pvalue is not None and res.adf_pvalue < 0.05
    assert res.kpss_pvalue is not None and res.kpss_pvalue > 0.05


def test_stationarity_tests_identify_random_walk():
    df = pd.DataFrame({"Y": _make_random_walk()})
    results = stationarity.run_stationarity_tests(df, ["Y"])
    res = results[0]
    assert res.adf_pvalue is not None and res.adf_pvalue > 0.1
    assert res.kpss_pvalue is not None and res.kpss_pvalue < 0.1


def test_stationarity_cli_outputs_csv(tmp_path: Path):
    n = 200
    df = pd.DataFrame(
        {
            "day": pd.date_range("2020-01-01", periods=n, freq="D"),
            "Y": _make_stationary_series(n=n, seed=99),
            "E_t": _make_stationary_series(n=n, seed=101),
        }
    )
    panel_path = tmp_path / "panel.parquet"
    df.to_parquet(panel_path)

    output_csv = tmp_path / "summary.csv"
    runner = CliRunner()
    result = runner.invoke(
        stationarity.main,
        [
            "--day-panel",
            str(panel_path),
            "--outcomes",
            "Y",
            "--exposure-column",
            "E_t",
            "--no-kpss",
            "--output",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_csv.exists()
    table = pd.read_csv(output_csv)
    expected_cols = {
        "column",
        "nobs",
        "adf_stat",
        "adf_pvalue",
        "adf_lags",
        "kpss_stat",
        "kpss_pvalue",
        "kpss_lags",
    }
    assert expected_cols == set(table.columns)
    assert "Y" in table["column"].values
