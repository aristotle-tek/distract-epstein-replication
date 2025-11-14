import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze import ardl


def _build_dataframe(e_series: np.ndarray, y_series: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=len(e_series), freq="D")
    return pd.DataFrame({"day": dates, "Y": y_series, "E": e_series})


def _simulate_related(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=1.0, size=n)
    y = np.zeros(n)
    noise = rng.normal(scale=0.1, size=n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.4 * e[t] + 0.2 * e[t - 1] + noise[t]
    return _build_dataframe(e, y)


def _simulate_unrelated(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=1.0, size=n)
    y = np.zeros(n)
    noise = rng.normal(scale=1.0, size=n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + noise[t]
    return _build_dataframe(e, y)


def _run_ardl(df: pd.DataFrame) -> ardl.ARDLResult:
    return ardl._run_single_outcome(
        df=df,
        y_col="Y",
        e_col="E",
        y_lags=1,
        e_lags=1,
        hac_lags=1,
        include_weekday=False,
        include_month=False,
    )


def test_ardl_detects_known_exposure_effect():
    df = _simulate_related(seed=42)
    result = _run_ardl(df)

    assert result.nobs == len(df) - 1
    assert result.beta_sum == pytest.approx(0.6, abs=0.1)
    assert result.p_sum_param < 0.01
    assert result.se_sum > 0

    coef_map = {detail.term: detail.coef for detail in result.lag_details}
    assert coef_map["E_lag0"] == pytest.approx(0.4, abs=0.1)
    assert coef_map["E_lag1"] == pytest.approx(0.2, abs=0.1)


def test_ardl_returns_high_pvalue_when_no_relationship():
    df = _simulate_unrelated(seed=123)
    result = _run_ardl(df)

    assert result.nobs == len(df) - 1
    assert result.beta_sum == pytest.approx(0.0, abs=0.1)
    assert result.p_sum_param > 0.1


def test_design_matrices_drop_wraparound_leads():
    n = 12
    df = _build_dataframe(
        e_series=np.arange(n, dtype=float),
        y_series=np.arange(n, dtype=float) * 0.0,
    )

    y, X, e_lag_cols, e_lead_cols, keep_idx = ardl._design_matrices(
        df=df,
        y_col="Y",
        e_col="E",
        y_lags=2,
        e_lags=1,
        include_weekday=False,
        include_month=False,
        e_leads=2,
    )

    assert e_lead_cols  # leads requested
    # exclude final rows so that no wrap-around appears in lead cols
    dropped = set(range(n)) - set(keep_idx)
    assert {n - 2, n - 1}.issubset(dropped)

    # Leading rows dropped because of lag requirements.
    assert {0, 1}.issubset(dropped)

    # Remaining sample should align with genuine forward-looking exposures
    last_idx = max(keep_idx)
    for h, col in enumerate(e_lead_cols, start=1):
        expected = df.loc[last_idx + h, "E"]
        assert X.loc[last_idx, col] == expected
