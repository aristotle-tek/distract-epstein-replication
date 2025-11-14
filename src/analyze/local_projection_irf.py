"""Local projection impulse response function estimation.

Example usage::

    python src/analyze/local_projection_irf.py run \
        --day-panel data_processed/day_panel.parquet \
        --outcomes N_t_z \
        --exposure-column daily_fox_epstein_density \
        --y-lags 7 --leads 5 --lags 14 \
        --hac-lags 7 --ri-iter 2000 \
        --report-name lp_irf_baseline
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import click
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import t as student_t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Palatino",
            "Palatino Linotype",
            "Book Antiqua",
            "URW Palladio L",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino",
        "mathtext.it": "Palatino:italic",
        "mathtext.bf": "Palatino:bold",
    }
)
if shutil.which("latex"):
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{mathpazo}"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths, run_metadata

DEFAULT_WINDOWS: List[tuple[int, int]] = [(0, 1), (0, 3), (0, 7), (0, 14), (-3, -1)]


@dataclass
class HorizonDesign:
    horizon: int
    y: pd.Series
    X: pd.DataFrame
    exposure_column: str
    sample_index: np.ndarray


@dataclass
class WindowDesign:
    start: int
    end: int
    label: str
    y: pd.Series
    X: pd.DataFrame
    exposure_column: str
    sample_index: np.ndarray


@dataclass
class HorizonResult:
    horizon: int
    nobs: int
    coef: float
    se: float
    t: float
    p: float
    ci_lower: float
    ci_upper: float
    df_resid: float
    uniform_lower: float | None = None
    uniform_upper: float | None = None



    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "horizon": int(self.horizon),
            "nobs": int(self.nobs),
            "coef": _finite_or_none(self.coef),
            "se": _finite_or_none(self.se),
            "t": _finite_or_none(self.t),
            "p": _finite_or_none(self.p),
            "ci_lower": _finite_or_none(self.ci_lower),
            "ci_upper": _finite_or_none(self.ci_upper),
            "uniform_lower": _finite_or_none(self.uniform_lower),
            "uniform_upper": _finite_or_none(self.uniform_upper),
            "df_resid": _finite_or_none(self.df_resid),
        }


@dataclass
class LocalProjectionResult:
    outcome: str
    exposure_column: str
    horizons: List[HorizonResult]
    y_lags: int
    leads: int
    lags: int
    hac_lags: int
    pointwise_crit: Dict[int, float]
    uniform_crit: float | None
    windows: List["WindowResult"] = field(default_factory=list)
    cumulative_windows: List["WindowResult"] = field(default_factory=list)
    pretrend_tests: "PretrendJointTest | None" = None

    def summary_dict(self) -> Dict[str, object]:
        return {
            "outcome": self.outcome,
            "exposure_column": self.exposure_column,
            "y_lags": int(self.y_lags),
            "leads": int(self.leads),
            "lags": int(self.lags),
            "hac_lags": int(self.hac_lags),
            "uniform_crit": None if self.uniform_crit is None else float(self.uniform_crit),
            "pointwise_crit": {int(h): float(val) for h, val in self.pointwise_crit.items()},
            "results": [h.to_dict() for h in self.horizons],
            "windows": [w.to_dict() for w in self.windows],
            "cumulative_windows": [w.to_dict() for w in self.cumulative_windows],
            "pretrend_tests": (
                None if self.pretrend_tests is None else self.pretrend_tests.to_dict()
            ),
        }


@dataclass
class WindowResult:
    start: int
    end: int
    label: str
    nobs: int
    coef: float
    se: float
    t: float
    p: float
    ci_lower: float
    ci_upper: float
    df_resid: float
    uniform_lower: float | None = None
    uniform_upper: float | None = None

    def to_dict(self) -> Dict[str, float | int | str | None]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
            "nobs": int(self.nobs),
            "coef": float(self.coef),
            "se": float(self.se),
            "t": float(self.t),
            "p": float(self.p),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "uniform_lower": (
                float(self.uniform_lower) if self.uniform_lower is not None else None
            ),
            "uniform_upper": (
                float(self.uniform_upper) if self.uniform_upper is not None else None
            ),
            "df_resid": float(self.df_resid),
        }


@dataclass
class DesignComponents:
    y_series: pd.Series
    base_parts: List[pd.DataFrame]


@dataclass
class PretrendJointTest:
    horizons: List[int]
    wald_stat: float | None = None
    wald_df: int | None = None
    wald_p: float | None = None
    observed_max_t: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "horizons": [int(h) for h in self.horizons],
            "wald_stat": None if self.wald_stat is None else float(self.wald_stat),
            "wald_df": None if self.wald_df is None else int(self.wald_df),
            "wald_p": None if self.wald_p is None else float(self.wald_p),
            "observed_max_t": (
                None if self.observed_max_t is None else float(self.observed_max_t)
            ),
        }


@dataclass
class FitStats:
    coef: float
    se: float
    t: float
    p: float
    ci_lower: float
    ci_upper: float
    df_resid: float
    nobs: int


def _finite_or_none(x: object) -> float | None:
    # sanitize (prevent NaN/Inf)
    if x is None:
        return None
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return None


def _prepare_design_components(
    df: pd.DataFrame,
    y_col: str,
    e_col: str,
    y_lags: int,
    include_weekday: bool,
    include_month: bool,
) -> DesignComponents:
    frame = df.copy()
    if "day" in frame.columns:
        frame = frame.sort_values("day").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    y_series = pd.to_numeric(frame[y_col], errors="coerce")
    e_series = pd.to_numeric(frame[e_col], errors="coerce")

    exposure_df = pd.DataFrame({e_col: e_series}).astype(float)

    lag_parts = {f"{y_col}_lag{i}": y_series.shift(i) for i in range(1, y_lags + 1)}
    lag_df = pd.DataFrame(lag_parts)
    if not lag_df.empty:
        lag_df = lag_df.apply(pd.to_numeric, errors="coerce").astype(float)

    calendar_df = _calendar_dummies(
        frame, include_weekday=include_weekday, include_month=include_month
    )
    if not calendar_df.empty:
        calendar_df = calendar_df.astype(float)

    base_parts: List[pd.DataFrame] = [exposure_df]
    if not lag_df.empty:
        base_parts.append(lag_df)
    if not calendar_df.empty:
        base_parts.append(calendar_df)

    return DesignComponents(y_series=y_series, base_parts=base_parts)


def _assemble_design(
    y_target: pd.Series, base_parts: Sequence[pd.DataFrame]
) -> tuple[pd.Series, pd.DataFrame, np.ndarray] | None:
    X = pd.concat([part.copy() for part in base_parts], axis=1)
    X["__target__"] = y_target
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna()
    if X.empty:
        return None

    y_vec = X.pop("__target__").astype(float)
    X = sm.add_constant(X, has_constant="add").astype(float)
    sample_idx = X.index.to_numpy(dtype=int)
    return y_vec, X, sample_idx


def _sum_shifted_series(series: pd.Series, start: int, end: int) -> pd.Series:
    total: pd.Series | None = None
    for h in range(start, end + 1):
        shifted = series.shift(-h)
        if total is None:
            total = shifted
        else:
            total = total.add(shifted, fill_value=np.nan)
    if total is None:
        total = pd.Series(index=series.index, dtype=float)
    return total


def _window_label(start: int, end: int) -> str:
    if start == end:
        return f"{start}"
    dash = "–"
    return f"{start}{dash}{end}"


def _calendar_dummies(df: pd.DataFrame, include_weekday: bool, include_month: bool) -> pd.DataFrame:
    cols: List[pd.DataFrame] = []
    if include_weekday and "weekday" in df.columns:
        cols.append(pd.get_dummies(df["weekday"].astype(int), prefix="wd", drop_first=True).astype(float))
    if include_month and "month" in df.columns:
        cols.append(pd.get_dummies(df["month"].astype(int), prefix="m", drop_first=True).astype(float))
    if "inaug_2025_onward" in df.columns:
        cols.append(df[["inaug_2025_onward"]].astype(float))
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.concat(cols, axis=1)


def _prepare_output_path(path: Path, no_clobber: bool) -> Path:
    if not no_clobber or not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path.stem
    suffix = path.suffix
    candidate = path.with_name(f"{stem}.{timestamp}{suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{stem}.{timestamp}_{counter}{suffix}")
        counter += 1
    return candidate


def _match_column(df: pd.DataFrame, name: str) -> str | None:
    lowered = name.lower()
    for col in df.columns:
        if col == name:
            return col
    for col in df.columns:
        if col.lower() == lowered:
            return col
    return None


def _resolve_outcomes(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    import fnmatch

    matched: List[str] = []
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if any(ch in pat for ch in "*?["):
            matched.extend(sorted(col for col in df.columns if fnmatch.fnmatch(col, pat)))
        elif pat in df.columns:
            matched.append(pat)
    deduped: List[str] = []
    seen: set[str] = set()
    for col in matched:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


def _build_designs(
    df: pd.DataFrame,
    y_col: str,
    e_col: str,
    y_lags: int,
    leads: int,
    lags: int,
    include_weekday: bool,
    include_month: bool,
) -> tuple[List[HorizonDesign], DesignComponents]:
    components = _prepare_design_components(
        df,
        y_col=y_col,
        e_col=e_col,
        y_lags=y_lags,
        include_weekday=include_weekday,
        include_month=include_month,
    )

    designs: List[HorizonDesign] = []
    for h in range(-leads, lags + 1):
        y_target = components.y_series.shift(-h)
        assembled = _assemble_design(y_target, components.base_parts)
        if assembled is None:
            continue
        y_vec, X, sample_idx = assembled
        if h < 0:
            k = -h
            lag_col = f"{y_col}_lag{k}"
            if lag_col in X.columns:
                X = X.drop(columns=[lag_col])
        designs.append(
            HorizonDesign(
                horizon=h,
                y=y_vec,
                X=X,
                exposure_column=e_col,
                sample_index=sample_idx,
            )
        )
    return designs, components


def _fit_projection(
    design: HorizonDesign | WindowDesign, hac_lags: int
) -> FitStats:
    model = sm.OLS(design.y, design.X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    coef = float(res.params[design.exposure_column])
    se = float(res.bse[design.exposure_column])
    t_val = coef / se if se and math.isfinite(se) and se != 0 else float("nan")
    p_series = res.pvalues
    if isinstance(p_series, pd.Series) and design.exposure_column in p_series.index:
        p_val = float(p_series[design.exposure_column])
    else:
        p_val = float("nan")
    df_resid = float(res.df_resid)
    if math.isfinite(df_resid) and df_resid > 0:
        crit = float(student_t.ppf(0.975, df_resid))
    else:
        crit = 1.96
    ci_lower = coef - crit * se if math.isfinite(se) else float("nan")
    ci_upper = coef + crit * se if math.isfinite(se) else float("nan")
    return FitStats(
        coef=coef,
        se=se,
        t=t_val,
        p=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        df_resid=df_resid,
        nobs=int(res.nobs),
    )


def _fit_horizon(design: HorizonDesign, hac_lags: int) -> HorizonResult:
    stats = _fit_projection(design, hac_lags=hac_lags)
    return HorizonResult(
        horizon=design.horizon,
        nobs=stats.nobs,
        coef=stats.coef,
        se=stats.se,
        t=stats.t,
        p=stats.p,
        ci_lower=stats.ci_lower,
        ci_upper=stats.ci_upper,
        df_resid=stats.df_resid,
    )


def _fit_window(design: WindowDesign, hac_lags: int) -> WindowResult:
    stats = _fit_projection(design, hac_lags=hac_lags)
    return WindowResult(
        start=design.start,
        end=design.end,
        label=design.label,
        nobs=stats.nobs,
        coef=stats.coef,
        se=stats.se,
        t=stats.t,
        p=stats.p,
        ci_lower=stats.ci_lower,
        ci_upper=stats.ci_upper,
        df_resid=stats.df_resid,
    )


def _cumulative_window_candidates(max_horizon: int) -> List[tuple[int, int]]:
    """Return windows (0,H) for H=1..max_horizon."""
    return [(0, h) for h in range(1, max_horizon + 1)]


def _build_window_designs(
    components: DesignComponents, e_col: str, windows: Sequence[tuple[int, int]]
) -> List[WindowDesign]:
    designs: List[WindowDesign] = []
    for start, end in windows:
        if start > end:
            continue
        y_target = _sum_shifted_series(components.y_series, start, end)
        assembled = _assemble_design(y_target, components.base_parts)
        if assembled is None:
            continue
        y_vec, X, sample_idx = assembled
        if end <= -1 and start < 0:
            drop_cols: List[str] = []
            for k in range(-end, -start + 1):
                col = f"{components.y_series.name}_lag{k}" if components.y_series.name else f"y_lag{k}"
                if col in X.columns:
                    drop_cols.append(col)
            if drop_cols:
                X = X.drop(columns=drop_cols)
        designs.append(
            WindowDesign(
                start=start,
                end=end,
                label=_window_label(start, end),
                y=y_vec,
                X=X,
                exposure_column=e_col,
                sample_index=sample_idx,
            )
        )
    return designs


def _apply_uniform_bands(result: LocalProjectionResult, uniform_cv: float | None) -> None:
    if uniform_cv is None:
        return
    for horizon_res in result.horizons:
        se = horizon_res.se
        if not math.isfinite(se):
            continue
        horizon_res.uniform_lower = horizon_res.coef - uniform_cv * se
        horizon_res.uniform_upper = horizon_res.coef + uniform_cv * se


def _apply_uniform_to_windows(
    window_results: Sequence[WindowResult], uniform_cv: float | None
) -> None:
    if uniform_cv is None:
        return
    for window_res in window_results:
        se = window_res.se
        if not math.isfinite(se):
            continue
        window_res.uniform_lower = window_res.coef - uniform_cv * se
        window_res.uniform_upper = window_res.coef + uniform_cv * se


def _wald_pretrend_test(
    horizon_results: Sequence[HorizonResult],
) -> tuple[float | None, int | None, float | None]:
    valid_pairs: List[tuple[float, float]] = []
    for res in horizon_results:
        coef = float(res.coef)
        se = float(res.se)
        if not math.isfinite(coef) or not math.isfinite(se) or se == 0:
            continue
        valid_pairs.append((coef, se))
    if not valid_pairs:
        return None, None, None
    stat = float(sum((coef / se) ** 2 for coef, se in valid_pairs))
    df = len(valid_pairs)
    p_val = float(1.0 - chi2.cdf(stat, df)) if math.isfinite(stat) else None
    return stat, df, p_val


def _observed_max_t(horizon_results: Sequence[HorizonResult]) -> float | None:
    t_values: List[float] = []
    for res in horizon_results:
        coef = float(res.coef)
        se = float(res.se)
        if not math.isfinite(coef) or not math.isfinite(se) or se == 0:
            continue
        t_abs = abs(coef / se)
        if math.isfinite(t_abs):
            t_values.append(t_abs)
    if not t_values:
        return None
    return float(max(t_values))


def _compute_pretrend_tests(
    horizon_results: Sequence[HorizonResult],
) -> PretrendJointTest | None:
    neg_h = sorted({res.horizon for res in horizon_results if res.horizon < 0})
    if not neg_h:
        return None
    filtered_results = [r for r in horizon_results if r.horizon in neg_h]

    observed_max = _observed_max_t(filtered_results)
    wald_stat, wald_df, wald_p = _wald_pretrend_test(filtered_results)

    return PretrendJointTest(
        horizons=neg_h,
        wald_stat=wald_stat,
        wald_df=wald_df,
        wald_p=wald_p,
        observed_max_t=observed_max,
    )

def _find_window(
    window_results: Sequence[WindowResult], start: int, end: int
) -> WindowResult | None:
    for res in window_results:
        if res.start == start and res.end == end:
            return res
    return None


def _format_estimate(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _format_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "—"
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return "—"
    return f"[{lower:.2f}, {upper:.2f}]"


def _build_key_metrics_rows(
    result: LocalProjectionResult,
    window_results: Sequence[WindowResult],
    ardl_metrics: Dict[str, float | str | None] | None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    metric_label = "ARDL: beta_sum"
    estimate = "NA"
    notes = "ARDL summary not provided"
    if ardl_metrics:
        q = ardl_metrics.get("e_lags")
        if q is not None and isinstance(q, (int, float)) and math.isfinite(q):
            metric_label = f"ARDL: beta_sum (q={int(q)})"
        beta_sum = ardl_metrics.get("beta_sum")
        if isinstance(beta_sum, (int, float)) and math.isfinite(beta_sum):
            estimate = _format_estimate(beta_sum)
        param_p = ardl_metrics.get("p_sum_param")
        if isinstance(param_p, (int, float)) and math.isfinite(param_p):
            notes = f"param p={param_p:.3f}"
        else:
            notes = ""
    rows.append(
        {
            "metric": metric_label,
            "estimate": estimate,
            "pointwise_CI": "—",
            "uniform_CI": "—",
            "notes": notes,
        }
    )

    note_base = f"HAC({result.hac_lags})"

    def add_window_row(start: int, end: int, label: str, extra_note: str = "") -> None:
        window = _find_window(window_results, start=start, end=end)
        if window is None:
            return
        uniform_ci = _format_interval(window.uniform_lower, window.uniform_upper)
        note = note_base
        if extra_note:
            note = f"{note}; {extra_note}" if note else extra_note
        rows.append(
            {
                "metric": label,
                "estimate": _format_estimate(window.coef),
                "pointwise_CI": _format_interval(window.ci_lower, window.ci_upper),
                "uniform_CI": uniform_ci,
                "notes": note,
            }
        )

    def cumulative_label(start: int, end: int) -> str:
        dash = "–"
        return f"LP: cumulative {start}{dash}{end}"

    add_window_row(0, 3, cumulative_label(0, 3))
    add_window_row(0, 7, cumulative_label(0, 7))
    lead_label = f"LP: leads ({-3} to {-1})"
    add_window_row(-3, -1, lead_label, extra_note="placebo")

    pretrend = result.pretrend_tests
    if pretrend is not None:
        if pretrend.wald_p is not None:
            rows.append(
                {
                    "metric": "LP: pre-trend Wald p-value",
                    "estimate": _format_estimate(pretrend.wald_p),
                    "pointwise_CI": "—",
                    "uniform_CI": "—",
                    "notes": (
                        f"chi^2({pretrend.wald_df}) = {_format_estimate(pretrend.wald_stat)}"
                        if pretrend.wald_stat is not None and pretrend.wald_df is not None
                        else ""
                    ),
                }
            )
        if pretrend.observed_max_t is not None:
            rows.append(
                {
                    "metric": "LP: pre-trend max |t|",
                    "estimate": _format_estimate(pretrend.observed_max_t),
                    "pointwise_CI": "—",
                    "uniform_CI": "—",
                    "notes": "",
                }
            )

    return rows


def _extract_ardl_metrics(
    payload: Dict[str, object] | None, outcome: str
) -> Dict[str, float | str | None] | None:
    if not payload:
        return None
    entry = payload.get(outcome)
    if entry is None:
        lowered = outcome.lower()
        for key, value in payload.items():
            if key.lower() == lowered:
                entry = value
                break
    if not isinstance(entry, dict):
        return None
    return {
        "beta_sum": entry.get("beta_sum"),
        "p_sum_param": entry.get("p_sum_param"),
        "e_lags": entry.get("e_lags"),
        "summary_note": entry.get("summary_note"),
    }


def _render_irf_axis(ax: plt.Axes, result: LocalProjectionResult) -> None:
    horizons = [h.horizon for h in result.horizons]
    betas = np.asarray([h.coef for h in result.horizons], dtype=float)
    ci_lower = np.asarray([h.ci_lower for h in result.horizons], dtype=float)
    ci_upper = np.asarray([h.ci_upper for h in result.horizons], dtype=float)
    uniform_lower = np.asarray(
        [np.nan if h.uniform_lower is None else h.uniform_lower for h in result.horizons],
        dtype=float,
    )
    uniform_upper = np.asarray(
        [np.nan if h.uniform_upper is None else h.uniform_upper for h in result.horizons],
        dtype=float,
    )

    line_color = "#134b94"
    ci_color = "#9ecae1"
    uniform_color = "#4f83cc"
    pre_period_color = "#E5E4E2" # "#e5efff"
    zero_marker_color = "#c4473a"

    pre_horizons = [h for h in horizons if h < 0]
    if pre_horizons:
        pre_min = min(pre_horizons) - 0.5
        ax.axvspan(pre_min, -0.5, color=pre_period_color, zorder=0)

    ax.axhline(0, color=line_color, linewidth=1.0, linestyle="--", alpha=0.65, zorder=1)
    ax.axvline(0, color=line_color, linewidth=1.0, linestyle="--", alpha=0.65, zorder=1)

    ax.fill_between(
        horizons,
        ci_lower,
        ci_upper,
        color=ci_color,
        alpha=0.45,
        label=r"$95\%$ HAC CI",
        zorder=2,
    )

    if np.isfinite(uniform_lower).any() and np.isfinite(uniform_upper).any():
        ax.fill_between(
            horizons,
            uniform_lower,
            uniform_upper,
            color=uniform_color,
            alpha=0.25,
            label=r"$95\%$ uniform band",
            zorder=3,
        )

    ax.plot(
        horizons,
        betas,
        color=line_color,
        linewidth=2.0,
        label="Estimate",
        zorder=4,
    )
    ax.scatter(horizons, betas, color=line_color, s=18, zorder=5)

    if 0 in horizons:
        zero_idx = horizons.index(0)
        zero_beta = betas[zero_idx]
        if np.isfinite(zero_beta):
            ax.scatter([0],[zero_beta], color=zero_marker_color, s=35, zorder=6)

        value_arrays = [betas, ci_lower, ci_upper]
        if np.isfinite(uniform_lower).any() and np.isfinite(uniform_upper).any():
            value_arrays.extend([uniform_lower, uniform_upper])
        finite_vals = np.concatenate(
            [arr[np.isfinite(arr)] for arr in value_arrays if np.isfinite(arr).any()]
        )
        if finite_vals.size:
            y_span = finite_vals.max() - finite_vals.min()
        else:
            y_span = 1.0
        if not math.isfinite(y_span) or y_span == 0:
            y_span = 1.0
        offset = 0.06 * y_span
        ax.annotate(
            f"{zero_beta:.2f}",
            xy=(0, zero_beta),
            xytext=(0.5, zero_beta + offset),
            textcoords="data",
            ha="left",
            va="bottom",
            fontsize=10,
            color=zero_marker_color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": line_color, "lw": 0.8},
            zorder=7,
        )

    unique_horizons = sorted(set(horizons))
    ax.set_xticks(unique_horizons)
    ax.set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Days relative to exposure shock")
    ax.set_ylabel("Outcome response (SD per 1 SD exposure shock)")
    ax.set_title("Event-study impulse response of novelty to Epstein attention")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _render_cumulative_axis(ax: plt.Axes, windows: Sequence[WindowResult]) -> None:
    relevant = [w for w in windows if w.start == 0 and w.end >= 0]
    relevant.sort(key=lambda w: w.end)
    if not relevant:
        ax.text(0.5, 0.5, "No cumulative windows available", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    horizons = np.array([w.end for w in relevant], dtype=int)
    coefs = np.asarray([w.coef for w in relevant], dtype=float)
    ci_lower = np.asarray([w.ci_lower for w in relevant], dtype=float)
    ci_upper = np.asarray([w.ci_upper for w in relevant], dtype=float)
    uniform_lower = np.asarray(
        [np.nan if w.uniform_lower is None else w.uniform_lower for w in relevant],
        dtype=float,
    )
    uniform_upper = np.asarray(
        [np.nan if w.uniform_upper is None else w.uniform_upper for w in relevant],
        dtype=float,
    )

    line_color = "#134b94"
    uniform_color = "#4f83cc"

    ax.axhline(0, color=line_color, linewidth=1.0, linestyle="--", alpha=0.65, zorder=1)

    mask_uniform = np.isfinite(uniform_lower) & np.isfinite(uniform_upper)
    if mask_uniform.any():
        ax.fill_between(
            horizons[mask_uniform],
            uniform_lower[mask_uniform],
            uniform_upper[mask_uniform],
            color=uniform_color,
            alpha=0.25,
            label=r"$95\%$ uniform band (windows)",
            zorder=2,
        )

    lower_err = np.where(np.isfinite(ci_lower), coefs - ci_lower, np.nan)
    upper_err = np.where(np.isfinite(ci_upper), ci_upper - coefs, np.nan)

    ax.errorbar(
        horizons,
        coefs,
        yerr=[lower_err, upper_err],
        fmt="-o",
        color=line_color,
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="Cumulative estimate",
        zorder=4,
    )

    ax.set_xlabel("End of cumulative window (days after shock)")
    ax.set_ylabel("Cumulative response (SD)")
    ax.set_xticks(horizons)
    ax.set_xlim(horizons.min() - 0.5, horizons.max() + 0.5)
    ax.set_title("Cumulative impulse response (0 to H)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_irf(result: LocalProjectionResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    _render_irf_axis(ax, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_irf_with_cumulative(
    result: LocalProjectionResult, path: Path, cumulative_windows: Sequence[WindowResult]
) -> Path:
    fig, (ax_irf, ax_cum) = plt.subplots(
        2,
        1,
        figsize=(8, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.2], "hspace": 0.35},
    )
    _render_irf_axis(ax_irf, result)
    _render_cumulative_axis(ax_cum, cumulative_windows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _write_outputs(
    result: LocalProjectionResult,
    window_results: Sequence[WindowResult],
    outdir: Path,
    no_clobber: bool,
    ardl_metrics: Dict[str, float | str | None] | None,
    design_details: Dict[str, object],
) -> Dict[str, Path]:
    csv_path = _prepare_output_path(outdir / f"lp_irf_{result.outcome}.csv", no_clobber=no_clobber)
    json_path = _prepare_output_path(outdir / f"lp_irf_{result.outcome}.json", no_clobber=no_clobber)
    figures_dir = paths.figures_mirror_for_outdir(outdir, ensure=True)
    plot_path = _prepare_output_path(
        figures_dir / f"lp_irf_{result.outcome}.png", no_clobber=no_clobber
    )
    key_metrics_path = _prepare_output_path(
        outdir / f"lp_irf_{result.outcome}_key_metrics.csv", no_clobber=no_clobber
    )
    window_path = _prepare_output_path(
        outdir / f"lp_irf_{result.outcome}_cumulative.csv", no_clobber=no_clobber
    )
    design_path = _prepare_output_path(
        outdir / f"lp_irf_{result.outcome}_design.json", no_clobber=no_clobber
    )

    df = pd.DataFrame([h.to_dict() for h in result.horizons])
    df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result.summary_dict(), fh, indent=2)

    key_rows = _build_key_metrics_rows(result, window_results, ardl_metrics)
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(key_metrics_path, index=False)

    window_df = pd.DataFrame([w.to_dict() for w in window_results])
    if not window_df.empty:
        window_df = window_df.copy()
        window_df["hac_lags"] = result.hac_lags
        window_df["uniform_crit"] = result.uniform_crit
    window_df.to_csv(window_path, index=False)

    with design_path.open("w", encoding="utf-8") as fh:
        json.dump(design_details, fh, indent=2)

    plot_combo_path = _prepare_output_path(
        figures_dir / f"lp_irf_{result.outcome}_with_cumulative.png", no_clobber=no_clobber
    )

    _plot_irf(result, plot_path)
    _plot_irf_with_cumulative(result, plot_combo_path, result.cumulative_windows)

    return {
        "csv": csv_path,
        "json": json_path,
        "plot": plot_path,
        "plot_with_cumulative": plot_combo_path,
        "key_metrics": key_metrics_path,
        "cumulative": window_path,
        "design": design_path,
    }


@click.group()
def cli() -> None:
    """Entry point for the local projection IRF CLI."""
    pass


@cli.command("run")
@click.option("--day-panel", "day_panel_path", type=click.Path(exists=True), required=True)
@click.option("--outcomes", "--outcome", "outcomes", type=str, required=True,
              help="Comma-separated list or globs of outcome columns.")
@click.option("--exposure-column", type=str, default=None, show_default=True,
              help="Column to use as E_t (default: 'E_t' or 'E_t_z').")
@click.option("--use-e-z/--no-use-e-z", default=True, show_default=True,
              help="Use standardized exposure column if available or create one.")
@click.option("--y-lags", type=int, default=7, show_default=True)
@click.option("--leads", type=int, default=5, show_default=True, help="Number of pre-trend horizons (h<0).")
@click.option("--lags", type=int, default=14, show_default=True, help="Number of post-shock horizons (h>=0).")
@click.option("--hac-lags", type=int, default=7, show_default=True)
@click.option("--include-weekday/--no-include-weekday", default=True, show_default=True)
@click.option("--include-month/--no-include-month", default=True, show_default=True)
@click.option(
    "--ardl-summary",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional ARDL summary JSON used to populate key metrics.",
)
@click.option("--report-name", type=str, default="lp_irf", show_default=True,
              help="Folder name under output_root/reports for artifacts.")
@click.option("--timestamp/--no-timestamp", default=False, show_default=True,
              help="Append timestamp suffix to report folder.")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Explicit output directory (overrides --report-name).")
@click.option("--no-clobber", is_flag=True, default=False, show_default=True)
@click.option("--tag", type=str, default="", show_default=True,
              help="Optional suffix for summary filename.")
def run_lp_irf(
    day_panel_path: str,
    outcomes: str,
    exposure_column: str | None,
    use_e_z: bool,
    y_lags: int,
    leads: int,
    lags: int,
    hac_lags: int,
    include_weekday: bool,
    include_month: bool,
    ardl_summary: Path | None,
    report_name: str,
    timestamp: bool,
    outdir: Path | None,
    no_clobber: bool,
    tag: str,
) -> None:
    if leads < 0 or lags < 0:
        raise click.ClickException("--leads and --lags must be non-negative integers.")

    if outdir is None:
        outdir_path = paths.report_run_dir(report_name, timestamp=timestamp, ensure=True)
    else:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(day_panel_path)
    if "day" in df.columns:
        df = df.sort_values("day").reset_index(drop=True)

    outcome_patterns = [s.strip() for s in outcomes.split(",") if s.strip()]
    if not outcome_patterns:
        raise click.ClickException("--outcomes resolved to an empty list.")
    outcome_columns = _resolve_outcomes(df, outcome_patterns)
    if not outcome_columns:
        raise click.ClickException("No outcome columns matched the provided patterns.")

    def _resolve_exposure_column() -> tuple[str, str | None]:
        base_e: str | None
        if exposure_column:
            base_e = _match_column(df, exposure_column)
            if base_e is None:
                raise click.ClickException(f"Exposure column '{exposure_column}' not found in day panel.")
        else:
            base_e = _match_column(df, "E_t")

        created: str | None = None

        if use_e_z:
            if base_e is None:
                # fall back to the common exposure column names
                resolved = _match_column(df, "E_t_z") or _match_column(df, "E_t")
                if resolved is None:
                    raise click.ClickException("Could not find exposure column.")
                base_e = resolved
            else:
                resolved = base_e

            if resolved == "E_t_z":
                z_name_candidates = ["E_t_z"]
            else:
                z_name_candidates = [f"Z_{resolved}", f"{resolved}_z", "E_t_z"]

            existing_z = next((matched for c in z_name_candidates if (matched := _match_column(df, c))), None)
            if existing_z:
                resolved = existing_z
            else:
                base_series = pd.to_numeric(df[resolved], errors="coerce")
                mean_val = base_series.mean(skipna=True)
                std_val = base_series.std(ddof=0, skipna=True)
                if not (std_val and math.isfinite(std_val) and std_val > 0):
                    raise click.ClickException("Exposure has zero/invalid std; cannot standardize.")
                zcol = f"Z_{resolved}"
                df[zcol] = (base_series - mean_val) / std_val
                resolved = zcol
                created = zcol
        else:
            resolved = base_e or _match_column(df, "E_t") or _match_column(df, "E_t_z")

        if resolved is None:
            raise click.ClickException("Failed to resolve exposure column.")

        return resolved, created

    resolved_e_col, created_z = _resolve_exposure_column()
    click.echo(f"(LP-IRF) Using exposure column: {resolved_e_col}")
    if created_z:
        click.echo(f"(LP-IRF) Created standardized exposure column '{created_z}'.")

    e_series = pd.to_numeric(df[resolved_e_col], errors="coerce").to_numpy()

    ardl_payload: Dict[str, object] | None = None
    if ardl_summary is not None:
        click.echo(f"(LP-IRF) Loading ARDL summary from {ardl_summary}")
        try:
            with Path(ardl_summary).open("r", encoding="utf-8") as fh:
                raw_payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Failed to parse ARDL summary JSON at {ardl_summary}"
            ) from exc
        if not isinstance(raw_payload, dict):
            raise click.ClickException(
                f"ARDL summary must be a JSON object: {ardl_summary}"
            )
        ardl_payload = {k: v for k, v in raw_payload.items() if k != "_meta"}

    combined_summary: Dict[str, Dict[str, object]] = {}

    for y_col in outcome_columns:
        click.echo(f"(LP-IRF) Outcome: {y_col}")
        designs, components = _build_designs(
            df=df,
            y_col=y_col,
            e_col=resolved_e_col,
            y_lags=y_lags,
            leads=leads,
            lags=lags,
            include_weekday=include_weekday,
            include_month=include_month,
        )
        if not designs:
            click.echo("  -> No observations available after lag/lead alignment.", err=True)
            continue

        designs_sorted = sorted(designs, key=lambda d: d.horizon)
        horizon_results: List[HorizonResult] = []
        pointwise_crit: Dict[int, float] = {}
        for design in designs_sorted:
            res = _fit_horizon(design, hac_lags=hac_lags)
            horizon_results.append(res)
            if math.isfinite(res.df_resid) and res.df_resid > 0:
                pointwise_crit[design.horizon] = float(student_t.ppf(0.975, res.df_resid))
            else:
                pointwise_crit[design.horizon] = 1.96
        horizon_results_sorted = sorted(horizon_results, key=lambda h: h.horizon)
        window_candidates = [
            window
            for window in DEFAULT_WINDOWS
            if window[0] >= -leads and window[1] <= lags
        ]
        window_designs = _build_window_designs(
            components, e_col=resolved_e_col, windows=window_candidates
        )
        window_results = [
            _fit_window(w_design, hac_lags=hac_lags) for w_design in window_designs
        ]
        window_results.sort(key=lambda w: (w.start, w.end))

        cumulative_candidates = _cumulative_window_candidates(lags)
        cumulative_window_designs = _build_window_designs(
            components, e_col=resolved_e_col, windows=cumulative_candidates
        )
        cumulative_window_results = [
            _fit_window(w_design, hac_lags=hac_lags) for w_design in cumulative_window_designs
        ]
        cumulative_window_results.sort(key=lambda w: (w.start, w.end))

        uniform_cv = None
        _apply_uniform_to_windows(window_results, uniform_cv)
        _apply_uniform_to_windows(cumulative_window_results, uniform_cv)

        pretrend_tests = _compute_pretrend_tests(
            horizon_results=horizon_results_sorted,
        )

        result = LocalProjectionResult(
            outcome=y_col,
            exposure_column=resolved_e_col,
            horizons=horizon_results_sorted,
            y_lags=y_lags,
            leads=leads,
            lags=lags,
            hac_lags=hac_lags,
            pointwise_crit=pointwise_crit,
            uniform_crit=uniform_cv,
            windows=list(window_results),
            cumulative_windows=list(cumulative_window_results),
            pretrend_tests=pretrend_tests,
        )

        _apply_uniform_bands(result, uniform_cv)

        ardl_metrics = _extract_ardl_metrics(ardl_payload, y_col)

        design_details = {
            "outcome": y_col,
            "day_panel": str(day_panel_path),
            "exposure_used": resolved_e_col,
            "exposure_argument": exposure_column,
            "use_e_z": use_e_z,
            "standardized_column_created": created_z,
            "y_lags": y_lags,
            "leads": leads,
            "lags": lags,
            "hac_lags": hac_lags,
            "include_weekday": include_weekday,
            "include_month": include_month,
            "uniform_crit": uniform_cv,
            "windows_requested": [
                {"start": start, "end": end} for start, end in window_candidates
            ],
            "cumulative_windows_requested": [
                {"start": start, "end": end} for start, end in cumulative_candidates
            ],
            "ardl_summary": str(ardl_summary) if ardl_summary else None,
        }

        outputs = _write_outputs(
            result,
            window_results,
            outdir_path,
            no_clobber=no_clobber,
            ardl_metrics=ardl_metrics,
            design_details=design_details,
        )

        summary_payload = result.summary_dict()
        combined_summary[y_col] = {
            "n_horizons": len(result.horizons),
            "uniform_crit": uniform_cv,
            "outputs": {k: str(v) for k, v in outputs.items()},
            "results": summary_payload.get("results", []),
            "windows": summary_payload.get("windows", []),
            "pointwise_crit": summary_payload.get("pointwise_crit", {}),
            "ardl": ardl_metrics,
            "design_details": design_details,
            "summary": summary_payload,
            "key_metrics_table": str(outputs.get("key_metrics")),
            "cumulative_table": str(outputs.get("cumulative")),
            "design_file": str(outputs.get("design")),
        }
        click.echo(
            "  -> saved outputs: "
            f"{outputs['csv']}, {outputs['json']}, {outputs['plot']}, "
            f"{outputs['plot_with_cumulative']}, {outputs['key_metrics']}, {outputs['cumulative']}"
        )

    if not combined_summary:
        raise click.ClickException("No outcome series were successfully estimated.")

    summary_name = f"lp_irf_summary_{tag}.json" if tag else "lp_irf_summary.json"
    summary_path = _prepare_output_path(outdir_path / summary_name, no_clobber=no_clobber)
    metadata = {
        "day_panel": str(day_panel_path),
        "exposure_argument": exposure_column,
        "exposure_used": resolved_e_col,
        "standardized_column_created": created_z,
        "use_e_z": use_e_z,
        "parameters": {
            "y_lags": y_lags,
            "leads": leads,
            "lags": lags,
            "hac_lags": hac_lags,
            "include_weekday": include_weekday,
            "include_month": include_month,
        },
        "results": combined_summary,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    click.echo(f"\nWrote summary: {summary_path}")

    run_metadata.record_run(
        ["local_projection_irf", Path(summary_path).stem],
        parameters={
            "y_lags": y_lags,
            "leads": leads,
            "lags": lags,
            "hac_lags": hac_lags,
            "use_e_z": use_e_z,
            "include_weekday": include_weekday,
            "include_month": include_month,
        },
        inputs={"day_panel": str(day_panel_path), "exposure": resolved_e_col, "outcomes": outcome_columns},
        outputs={"summary": str(summary_path)},
    )


if __name__ == "__main__":
    cli()
