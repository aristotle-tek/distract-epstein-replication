import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze import local_projection_irf as lp


def _build_dataframe(e_series: np.ndarray, y_series: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=len(e_series), freq="D")
    return pd.DataFrame({"day": dates, "Y": y_series, "E": e_series})


def _simulate_irf(seed: int = 0, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=1.0, size=n)
    noise = rng.normal(scale=0.05, size=n)
    coefs = {0: 0.6, 1: 0.3, 2: -0.1}
    y = np.zeros(n)
    for t in range(n):
        value = noise[t]
        for lag, coef in coefs.items():
            if t - lag >= 0:
                value += coef * e[t - lag]
        y[t] = value
    return _build_dataframe(e, y)


def _simulate_unrelated(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=1.0, size=n)
    y = np.zeros(n)
    noise = rng.normal(scale=1.0, size=n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + noise[t]
    return _build_dataframe(e, y)


def _collect_horizon_results(
    df: pd.DataFrame,
    *,
    y_lags: int,
    leads: int,
    lags: int,
    hac_lags: int,
) -> tuple[dict[int, lp.HorizonResult], lp.DesignComponents]:
    designs, components = lp._build_designs(
        df=df,
        y_col="Y",
        e_col="E",
        y_lags=y_lags,
        leads=leads,
        lags=lags,
        include_weekday=False,
        include_month=False,
    )
    horizon_results = {}
    for design in designs:
        result = lp._fit_horizon(design, hac_lags=hac_lags)
        horizon_results[design.horizon] = result
    return horizon_results, components


def test_local_projection_detects_impulse_response():
    df = _simulate_irf(seed=42)
    horizons, components = _collect_horizon_results(
        df,
        y_lags=0,
        leads=0,
        lags=2,
        hac_lags=1,
    )

    assert set(horizons) == {0, 1, 2}

    assert horizons[0].coef == pytest.approx(0.6, abs=0.05)
    assert horizons[1].coef == pytest.approx(0.3, abs=0.05)
    assert horizons[2].coef == pytest.approx(-0.1, abs=0.05)

    window_designs = lp._build_window_designs(
        components,
        e_col="E",
        windows=[(0, 2)],
    )
    assert len(window_designs) == 1
    window_result = lp._fit_window(window_designs[0], hac_lags=1)
    assert window_result.coef == pytest.approx(0.6 + 0.3 - 0.1, abs=0.08)


def test_local_projection_returns_high_pvalue_when_unrelated():
    df = _simulate_unrelated(seed=7)
    horizons, _ = _collect_horizon_results(
        df,
        y_lags=1,
        leads=0,
        lags=1,
        hac_lags=1,
    )

    # Horizon 0 should align with contemporaneous relationship
    horizon0 = horizons[0]
    assert horizon0.nobs == len(df) - 1
    assert horizon0.coef == pytest.approx(0.0, abs=0.05)
    assert horizon0.p > 0.1
