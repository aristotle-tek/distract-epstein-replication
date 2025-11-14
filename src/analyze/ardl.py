"""
Run ARDL models

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import click
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t, norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running via `python src/...`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths, run_metadata


@dataclass
class LagDetail:
    lag_index: int
    term: str
    coef: float
    se_hac: float
    t: float
    p_hac: float

    def to_dict(self) -> Dict[str, float | int | None]:
        data: Dict[str, float | int | None] = {
            "lag_index": int(self.lag_index),
            "term": self.term,
            "coef": float(self.coef),
            "se_hac": float(self.se_hac),
            "t": float(self.t),
            "p_hac": float(self.p_hac),
        }
        return data


@dataclass
class ARDLResult:
    outcome: str
    lag_details: List[LagDetail]
    beta_sum: float
    se_sum: float
    t_sum: float
    p_sum_param: float
    nobs: int
    r2: float
    start_date: str
    end_date: str
    y_lags: int
    e_lags: int
    hac_lags: int
    summary_note: str
    beta_leads_sum: float | None = None
    se_leads_sum: float | None = None
    t_leads_sum: float | None = None
    p_leads_param: float | None = None
    # for plotting/inference consistency
    use_t: bool = True
    df_resid: float | None = None


def _e_lead_cols(arr_E: np.ndarray, L: int, base_name: str) -> Tuple[List[str], np.ndarray]:
    # lead h means Et+ h → np.roll with negative shift
    names = [f"{base_name}_lead{h}" for h in range(1, L + 1)]
    mat = np.column_stack([np.roll(arr_E, -h) for h in range(1, L + 1)])
    return names, mat


def _calendar_dummies(df, include_weekday=True, include_month=True):
    cols = []
    if include_weekday and "weekday" in df.columns:
        cols.append(pd.get_dummies(df["weekday"].astype(int), prefix="wd", drop_first=True).astype(float))
    if include_month and "month" in df.columns:
        cols.append(pd.get_dummies(df["month"].astype(int), prefix="m", drop_first=True).astype(float))
    if "inaug_2025_onward" in df.columns:
        cols.append(df[["inaug_2025_onward"]].astype(float))
    return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=df.index)


def _lag_cols(arr: np.ndarray, lags: int, base_name: str) -> Tuple[List[str], np.ndarray]:
    names = [f"{base_name}_lag{i}" for i in range(1, lags + 1)]
    mat = np.column_stack([np.roll(arr, i) for i in range(1, lags + 1)])
    return names, mat


def _e_lag_cols(arr_E: np.ndarray, q: int, base_name: str) -> Tuple[List[str], np.ndarray]:
    names = [f"{base_name}_lag0"] + [f"{base_name}_lag{j}" for j in range(1, q + 1)]
    mat = np.column_stack([np.roll(arr_E, j) for j in range(0, q + 1)])
    return names, mat


def _trim_for_true_lags(
    df: pd.DataFrame, y_lags: int, e_lags: int, e_leads: int = 0
) -> pd.Index:
    # drop first max(y_lags,e_lags) + last e_leads (if any) to avoid look-ahead from circular construction
    n = len(df.index)
    start = min(max(y_lags, e_lags), n)
    end = n - e_leads if e_leads > 0 else n
    end = max(start, end)
    return df.index[start:end]


def _fit_hac_ols(y: np.ndarray, X: np.ndarray, hac_lags: int):
    model = sm.OLS(np.asarray(y, float), np.asarray(X, float))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def _wald_sum_from_result(
    res_sm, param_series: pd.Series, cov: pd.DataFrame, e_cols: List[str]
) -> Tuple[float, float, float, float]:
    # Wald test for H0: sum(beta_j)=0 using R * beta = 0 with R = [1 ... 1] on the E-lag block
    beta_sum = float(param_series[e_cols].sum())
    w = np.ones(len(e_cols))
    cov_E = cov.loc[e_cols, e_cols].to_numpy()
    var_sum = float(w @ cov_E @ w)
    se_sum = math.sqrt(var_sum) if var_sum > 0 else float("nan")
    t_sum = beta_sum / se_sum if se_sum > 0 else float("nan")

    R = np.zeros((1, len(param_series)), dtype=float)
    col_index = {name: i for i, name in enumerate(param_series.index)}
    for name in e_cols:
        R[0, col_index[name]] = 1.0

    # use statsmodels robust Wald with the model's chosen small-sample behavior (chi2 /F as appropriate)
    wald = res_sm.wald_test(R)
    p_sum = float(wald.pvalue)
    return beta_sum, se_sum, t_sum, p_sum


def _design_matrices(
    df: pd.DataFrame,
    y_col: str,
    e_col: str,
    y_lags: int,
    e_lags: int,
    include_weekday: bool,
    include_month: bool,
    include_trend: bool = False,
    e_leads: int = 0,
) -> Tuple[pd.Series, pd.DataFrame, List[str], List[str], pd.Index]:
    df = df.copy()
    if "day" in df.columns:
        df = df.sort_values("day").reset_index(drop=True)

    y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    e_raw = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)

    ylag_names, ylag_mat = _lag_cols(y_raw, y_lags, y_col)
    e_lag_names, e_lag_mat = _e_lag_cols(e_raw, e_lags, e_col)
    parts = [
        pd.DataFrame(ylag_mat, columns=ylag_names),
        pd.DataFrame(e_lag_mat, columns=e_lag_names),
    ]

    e_lead_names: List[str] = []
    if e_leads and e_leads > 0:
        e_lead_names, e_lead_mat = _e_lead_cols(e_raw, e_leads, e_col)
        parts.append(pd.DataFrame(e_lead_mat, columns=e_lead_names))

    Xc = _calendar_dummies(df, include_weekday=include_weekday, include_month=include_month)
    if not Xc.empty:
        parts.append(Xc.astype(float))

    control_cols = []
    if include_trend and "trend" in df.columns:
        trend_series = pd.to_numeric(df["trend"], errors="coerce")
        if not pd.isna(trend_series).all():
            control_cols.append(trend_series.astype(float).rename("trend"))
    for col in [
        "V_t_z",
        "novelty_posts_z",
        "novelty_ref_posts_z",
        "novelty_low_sample_flag",
        "novelty_missing_flag",
        "no_posts_flag",
    ]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if not pd.isna(series).all():
                control_cols.append(series.rename(col))
    if control_cols:
        parts.append(pd.concat(control_cols, axis=1))

    X = pd.concat(parts, axis=1)
    y = pd.Series(y_raw, name=y_col)

    keep_idx = _trim_for_true_lags(df, y_lags=y_lags, e_lags=e_lags, e_leads=e_leads)
    X = X.loc[keep_idx]
    y = y.loc[keep_idx]

    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    X = sm.add_constant(X, has_constant="add")
    return y, X, e_lag_names, e_lead_names, y.index


def _prepare_output_path(path: Path, no_clobber: bool) -> Path:
    if not no_clobber:
        return path
    if not path.exists():
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


def _build_lag_detail_rows(res: ARDLResult) -> List[Dict[str, float | int | None]]:
    rows: List[Dict[str, float | int | None]] = []
    for detail in res.lag_details:
        rows.append(detail.to_dict())

    sum_row: Dict[str, float | int | None] = {
        "term": "E_sum",
        "coef": float(res.beta_sum),
        "se_hac": float(res.se_sum),
        "t": float(res.t_sum),
        "p_hac": float(res.p_sum_param),
    }
    rows.append(sum_row)

    if res.beta_leads_sum is not None:
        leads_row = {
            "term": "E_leads_sum",
            "coef": float(res.beta_leads_sum),
            "se_hac": float(res.se_leads_sum),
            "t": float(res.t_leads_sum),
            "p_hac": float(res.p_leads_param) if res.p_leads_param is not None else float("nan"),
        }
        rows.append(leads_row)

    return rows


def _write_lag_detail_csv(res: ARDLResult, outdir: Path, no_clobber: bool) -> Path:
    detail_rows = _build_lag_detail_rows(res)
    detail_df = pd.DataFrame(detail_rows)

    detail_path = _prepare_output_path(
        outdir / f"ardl_{res.outcome}_E_lags_detail.csv", no_clobber=no_clobber
    )
    detail_df.to_csv(detail_path, index=False)
    return detail_path


def _save_lag_stem_plot(res: ARDLResult, outdir: Path, no_clobber: bool) -> Path | None:
    if not res.lag_details:
        return None

    lag_indices = np.array([detail.lag_index for detail in res.lag_details], dtype=float)
    betas = np.array([detail.coef for detail in res.lag_details], dtype=float)
    ses = np.array([detail.se_hac for detail in res.lag_details], dtype=float)

    # critical val aligned with statsmodels robust inference.
    if res.use_t and res.df_resid is not None and math.isfinite(res.df_resid):
        crit = float(student_t.ppf(0.975, res.df_resid))
    else:
        crit = float(norm.ppf(0.975))
    ci = crit * ses

    fig, ax = plt.subplots(figsize=(8, 5))
    stem_kwargs = {"basefmt": " "}
    try:
        markerline, stemlines, baseline = ax.stem(
            lag_indices, betas, use_line_collection=True, **stem_kwargs
        )
    except TypeError:
        markerline, stemlines, baseline = ax.stem(lag_indices, betas, **stem_kwargs)
    plt.setp(markerline, color="#1f77b4")
    plt.setp(stemlines, color="#1f77b4")

    if np.isfinite(ci).any():
        ax.errorbar(
            lag_indices,
            betas,
            yerr=ci,
            fmt="none",
            ecolor="#444444",
            elinewidth=1.0,
            capsize=4,
        )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xticks(lag_indices)
    ax.set_xticklabels([str(int(i)) for i in lag_indices])
    ax.set_xlabel("Lag index")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"{res.outcome}: Epstein lag coefficients")

    if res.beta_sum is not None and math.isfinite(res.beta_sum):
        annotation = f"Sum = {res.beta_sum:.3f}"
    else:
        annotation = "Sum = nan"
    ax.text(
        0.98,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()

    figures_dir = paths.figures_mirror_for_outdir(outdir, ensure=True)
    plot_path = _prepare_output_path(
        figures_dir / f"ardl_{res.outcome}_E_lags_betas.png", no_clobber=no_clobber
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _run_single_outcome(
    df: pd.DataFrame,
    y_col: str,
    e_col: str,
    y_lags: int,
    e_lags: int,
    hac_lags: int,
    include_weekday: bool,
    include_month: bool,
    include_trend: bool = False,
    e_leads: int = 0,
) -> ARDLResult:
    y, X, e_lag_cols, e_lead_cols, keep_idx = _design_matrices(
        df,
        y_col,
        e_col,
        y_lags,
        e_lags,
        include_weekday,
        include_month,
        include_trend=include_trend,
        e_leads=e_leads,
    )

    res_sm = _fit_hac_ols(y.to_numpy(dtype=float), X.to_numpy(dtype=float), hac_lags=hac_lags)
    params = pd.Series(res_sm.params, index=X.columns)
    bse    = pd.Series(res_sm.bse,    index=X.columns)
    tvals  = pd.Series(res_sm.tvalues,index=X.columns)
    pvals  = pd.Series(res_sm.pvalues,index=X.columns)
    cov_df = pd.DataFrame(res_sm.cov_params(), index=X.columns, columns=X.columns)
    use_t = bool(getattr(res_sm, "use_t", True))

    # Sum over E lags via statsmodels Wald test
    beta_sum, se_sum, t_sum, p_sum_param = _wald_sum_from_result(res_sm, params, cov_df, e_lag_cols)

    beta_leads_sum = se_leads_sum = t_leads_sum = p_leads_param = None
    if e_lead_cols:
        beta_leads_sum, se_leads_sum, t_leads_sum, p_leads_param = _wald_sum_from_result(
            res_sm, params, cov_df, e_lead_cols
        )

    start_date = str(df.loc[y.index.min(), "day"]) if "day" in df.columns and len(y) else ""
    end_date   = str(df.loc[y.index.max(), "day"]) if "day" in df.columns and len(y) else ""

    lag_details: List[LagDetail] = []
    for idx, term in enumerate(e_lag_cols):
        lag_details.append(
            LagDetail(
                lag_index=idx,
                term=term,
                coef=float(params[term]),
                se_hac=float(bse[term]),
                t=float(tvals[term]),
                p_hac=float(pvals[term]),
            )
        )

    summary_note = (
        f"Short-run lag sum (E_sum) = {beta_sum:.4f} ± {se_sum:.4f} "
        f"(HAC p = {('nan' if not math.isfinite(p_sum_param) else f'{p_sum_param:.3f}')})"
    )

    return ARDLResult(
        outcome=y_col,
        lag_details=lag_details,
        beta_sum=float(beta_sum),
        se_sum=float(se_sum),
        t_sum=float(t_sum),
        p_sum_param=float(p_sum_param),
        summary_note=summary_note,
        beta_leads_sum=None if beta_leads_sum is None else float(beta_leads_sum),
        se_leads_sum=None if se_leads_sum is None else float(se_leads_sum),
        t_leads_sum=None if t_leads_sum is None else float(t_leads_sum),
        p_leads_param=None if p_leads_param is None else float(p_leads_param),
        nobs=int(res_sm.nobs),
        r2=float(res_sm.rsquared),
        start_date=start_date,
        end_date=end_date,
        y_lags=y_lags,
        e_lags=e_lags,
        hac_lags=hac_lags,
        use_t=use_t,
        df_resid=float(res_sm.df_resid),
    )


def _write_per_outcome_outputs(
    res: ARDLResult, outdir: Path, no_clobber: bool
) -> Tuple[Path, Path, Path, Path | None]:
    rows = []
    for detail in res.lag_details:
        rows.append(detail.to_dict())
    row_sum = {
        "term": "E_sum",
        "lag_index": None,
        "coef": res.beta_sum,
        "se_hac": res.se_sum,
        "t": res.t_sum,
        "p_hac": res.p_sum_param,
    }
    rows.append(row_sum)
    if res.beta_leads_sum is not None:
        rows.append(
            {
                "term": "E_leads_sum",
                "lag_index": None,
                "coef": res.beta_leads_sum,
                "se_hac": res.se_leads_sum,
                "t": res.t_leads_sum,
                "p_hac": res.p_leads_param,
            }
        )

    df_out = pd.DataFrame(rows)
    csv_path = _prepare_output_path(outdir / f"ardl_{res.outcome}.csv", no_clobber=no_clobber)
    json_path = _prepare_output_path(outdir / f"ardl_{res.outcome}.json", no_clobber=no_clobber)
    df_out.to_csv(csv_path, index=False)

    detail_path = _write_lag_detail_csv(res, outdir, no_clobber=no_clobber)
    plot_path = _save_lag_stem_plot(res, outdir, no_clobber=no_clobber)

    summary = {
        "outcome": res.outcome,
        "nobs": res.nobs,
        "r2": res.r2,
        "start": res.start_date,
        "end": res.end_date,
        "y_lags": res.y_lags,
        "e_lags": res.e_lags,
        "hac_lags": res.hac_lags,
        "summary_note": res.summary_note,
        "beta_sum": res.beta_sum,
        "se_sum": res.se_sum,
        "t_sum": res.t_sum,
        "p_sum_param": res.p_sum_param,
        "lag_details": [detail.to_dict() for detail in res.lag_details],
    }
    if res.beta_leads_sum is not None:
        summary.update(
            {
                "beta_leads_sum": res.beta_leads_sum,
                "se_leads_sum": res.se_leads_sum,
                "t_leads_sum": res.t_leads_sum,
                "p_leads_param": res.p_leads_param,
            }
        )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path, detail_path, plot_path


@click.group()
def cli():
    pass


@cli.command("run")
@click.option("--day-panel", "day_panel_path", type=click.Path(exists=True), required=True)
@click.option(
    "--outcomes",
    type=str,
    required=True,
    help="Comma-separated list of outcome columns, e.g. 'N_t_z'",
)
@click.option("--y-lags", type=int, default=7, show_default=True)
@click.option("--e-lags", type=int, default=3, show_default=True)
@click.option(
    "--exposure-column",
    type=str,
    default=None,
    show_default=True,
    help="Exposure column to use as E_t (case-insensitive; default 'E_t').",
)
@click.option("--use-e-z/--no-use-e-z", default=True, show_default=True)
@click.option("--hac-lags", type=int, default=7, show_default=True)
@click.option("--include-weekday/--no-include-weekday", default=True, show_default=True)
@click.option("--include-month/--no-include-month", default=True, show_default=True)
@click.option(
    "--include-trend/--no-include-trend",
    default=False,
    show_default=True,
    help="Include a deterministic time trend as an additional control column.",
)
@click.option("--report-name", type=str, default="ardl", show_default=True,
              help="Folder name under output_root/reports/ for this run.")
@click.option("--timestamp/--no-timestamp", default=False, show_default=True,
              help="Append a timestamp suffix to the report folder.")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Explicit output directory (overrides --report-name).")
@click.option("--e-leads", type=int, default=0, show_default=True,
              help="Add Et leads 1..L as regressors (placebo).")
@click.option("--no-clobber", is_flag=True, default=False, show_default=True,
              help="When enabled, append .<YYYYMMDD_HHMMSS> before overwriting existing outputs.")
@click.option("--tag", type=str, default="", show_default=True,
              help="If set, write ardl_summary_<tag>.json instead of ardl_summary.json.")
@click.option("--placebo-pattern", type=str, show_default=True,
              help="Glob pattern(s) (comma-separated) describing placebo candidates.")
def run_ardl(
    day_panel_path: str,
    outcomes: str,
    y_lags: int,
    e_lags: int,
    exposure_column: str | None,
    use_e_z: bool,
    hac_lags: int,
    include_weekday: bool,
    include_month: bool,
    include_trend: bool,
    report_name: str,
    timestamp: bool,
    outdir: Path | None,
    e_leads: int,
    no_clobber: bool,
    tag: str,
    placebo_pattern: str,
):
    if outdir is None:
        outdir_path = paths.report_run_dir(report_name, timestamp=timestamp, ensure=True)
    else:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)

    day_panel = pd.read_parquet(day_panel_path)
    if "day" in day_panel.columns:
        day_panel = day_panel.sort_values("day").reset_index(drop=True)

    day_panel["trend"] = np.arange(len(day_panel), dtype=float)

    df = day_panel

    def _match_column(name: str) -> str | None:
        lowered = name.lower()
        for col in df.columns:
            if col == name:
                return col
        for col in df.columns:
            if col.lower() == lowered:
                return col
        return None

    base_e_col: str | None
    if exposure_column:
        base_e_col = _match_column(exposure_column)
        if base_e_col is None:
            raise click.ClickException(
                f"Exposure column '{exposure_column}' not found in {day_panel_path}."
            )
    else:
        base_e_col = _match_column("E_t")

    resolved_e_col: str | None = None
    created_z_col: str | None = None

    if base_e_col is None:
        if use_e_z:
            resolved_e_col = _match_column("E_t_z")
            base_e_col = resolved_e_col
        if resolved_e_col is None:
            raise click.ClickException(
                "Could not find Epstein series col ('E_t' or 'E_t_z') in the day panel."
            )

    if use_e_z:
        if base_e_col is None:
            raise click.ClickException("Exposure col resolution failed; no base col available.")
        z_candidates = []
        if base_e_col == "E_t":
            z_candidates.append("E_t_z")
        else:
            z_candidates.extend([
                f"Z_{base_e_col}",
                f"z_{base_e_col}",
                f"{base_e_col}_z",
            ])
        for candidate in z_candidates:
            match = _match_column(candidate)
            if match is not None:
                resolved_e_col = match
                break

        if resolved_e_col is None:
            base_series = pd.to_numeric(df[base_e_col], errors="coerce")
            mean_val = base_series.mean(skipna=True)
            std_val = base_series.std(ddof=0, skipna=True)
            if std_val and math.isfinite(std_val) and std_val > 0:
                resolved_name = "E_t_z" if base_e_col == "E_t" else f"Z_{base_e_col}"
                df[resolved_name] = (base_series - mean_val) / std_val
                resolved_e_col = resolved_name
                created_z_col = resolved_name
            else:
                click.echo(
                    f"(ARDL) Warning: could not standardize exposure col '{base_e_col}'. "
                    "Reverting to raw values.",
                    err=True,
                )
                resolved_e_col = base_e_col
    else:
        resolved_e_col = base_e_col

    if resolved_e_col is None or resolved_e_col not in df.columns:
        raise click.ClickException(
            f"Exposure column resolution failed; final col '{resolved_e_col}' unavailable."
        )

    click.echo(
        f"(ARDL) Exposure base col: {base_e_col}; using col: {resolved_e_col}"
    )
    if created_z_col:
        click.echo(f"(ARDL) Created standardized col '{created_z_col}'.")

    e_col = resolved_e_col

    y_cols = [s.strip() for s in outcomes.split(",") if s.strip()]
    if not y_cols:
        raise click.ClickException("No outcome columns were provided.")

    combined_summary = {}
    for y_col in y_cols:
        click.echo(f"(ARDL) outcome: {y_col}  (E: {e_col})")
        res = _run_single_outcome(
            df=df,
            y_col=y_col,
            e_col=e_col,
            y_lags=y_lags,
            e_lags=e_lags,
            hac_lags=hac_lags,
            include_weekday=include_weekday,
            include_month=include_month,
            include_trend=include_trend,
            e_leads=e_leads,
        )

        csv_path, json_path, detail_path, plot_path = _write_per_outcome_outputs(
            res, outdir_path, no_clobber=no_clobber
        )
        entry = {
            "beta_sum": res.beta_sum,
            "se_sum": res.se_sum,
            "p_sum_param": res.p_sum_param,
            "nobs": res.nobs,
            "r2": res.r2,
            "start": res.start_date,
            "end": res.end_date,
            "summary_note": res.summary_note,
        }
        if res.beta_leads_sum is not None:
            entry.update(
                {
                    "beta_leads_sum": res.beta_leads_sum,
                    "se_leads_sum": res.se_leads_sum,
                    "p_leads_param": res.p_leads_param,
                }
            )
        combined_summary[y_col] = entry
        click.echo(
            f"  -> sum(E lags) = {res.beta_sum:.4f} (se={res.se_sum:.4f}, p_param={res.p_sum_param:.3f}); n={res.nobs}, R2={res.r2:.3f}"
        )
        if res.beta_leads_sum is not None:
            click.echo(
                f"     lead-sum = {res.beta_leads_sum:.4f} (se={res.se_leads_sum:.4f}, p_param={res.p_leads_param:.3f})"
            )
        saved_bits = [str(csv_path), str(json_path)]
        if detail_path is not None:
            saved_bits.append(str(detail_path))
        if plot_path is not None:
            saved_bits.append(str(plot_path))
        click.echo(f"  Saved: {' ; '.join(saved_bits)}")

    combined_summary["_meta"] = {
        "day_panel": str(day_panel_path),
        "exposure_argument": exposure_column,
        "exposure_base_column": base_e_col,
        "exposure_column_used": e_col,
        "use_e_z": use_e_z,
        "standardized_column_created": created_z_col,
        "include_trend": include_trend,
    }

    fname = f"ardl_summary_{tag}.json" if tag else "ardl_summary.json"
    summary_path = _prepare_output_path(outdir_path / fname, no_clobber=no_clobber)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    click.echo(f"\nWrote combined summary: {summary_path}")


if __name__ == "__main__":
    cli()
