#!/usr/bin/env python3
"""
Consolidates the outputs and generates plots and laTeX tables
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence
import sys

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths



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


LOGGER = logging.getLogger("publication_outputs")


@dataclass(frozen=True)
class SpecKey:
    # Identifier for a row in ``ardl_results_compiled.csv``

    report: str
    label: str
    outcome: str = "N_t_z"
    tag: str | None = None
    required: bool = True


def _read_compiled_results(compiled_path: Path | None) -> pd.DataFrame:
    if compiled_path is None:
        compiled_path = paths.tables_dir(ensure=False) / "ardl_results_compiled.csv"
    if not compiled_path.exists():
        raise click.ClickException(
            "Compiled ARDL results not found. Run src/compile_ardl_results.py first."
        )
    LOGGER.info("Loading compiled ARDL results from %s", compiled_path)
    df = pd.read_csv(compiled_path)
    required_cols = {
        "report",
        "tag",
        "outcome",
        "beta_sum",
        "se_sum",
        "p_sum_param",
        "nobs",
        "r2",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise click.ClickException(
            f"Compiled results are missing columns: {', '.join(sorted(missing_cols))}"
        )
    return df


def _ensure_reports_dir(reports_dir: Path | None) -> Path:
    if reports_dir is None:
        reports_dir = paths.reports_dir(ensure=False)
    if not reports_dir.exists():
        raise click.ClickException(
            f"Reports directory does not exist. Expected outputs under: {reports_dir}"
        )
    return reports_dir


def _tables_dir() -> Path:
    return paths.tables_dir(ensure=True)


def _figures_dir() -> Path:
    return paths.figures_dir(ensure=True)


def _format_decimal(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def _format_integer(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{int(round(float(value))):d}"


def _format_p(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_sample(start: str | float | None, end: str | float | None) -> str:
    def _fmt(value: str | float | None) -> str:
        if value is None:
            return "--"
        if isinstance(value, float) and not np.isfinite(value):
            return "--"
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return str(value)
        return ts.strftime("%Y-%m-%d")

    start_str = _fmt(start)
    end_str = _fmt(end)
    if start_str == "--" and end_str == "--":
        return "--"
    return rf"{start_str}--{end_str}"


def _uniform_excludes_zero(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None or not (np.isfinite(lower) and np.isfinite(upper)):
        return "--"
    return "Yes" if (lower > 0 and upper > 0) or (lower < 0 and upper < 0) else "No"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_lp_result(
    reports_dir: Path,
    report_name: str = "lp_irf_baseline",
    outcome: str = "N_t_z",
) -> dict | None:
    summary_path = reports_dir / report_name / "lp_irf_summary.json"
    if not summary_path.exists():
        LOGGER.warning(
            "LP summary not found at %s; skipping related appendix tables.", summary_path
        )
        return None
    payload = _load_json(summary_path)
    results = payload.get("results", {})
    entry = results.get(outcome)
    if entry is None:
        LOGGER.warning(
            "Outcome '%s' missing from LP summary %s; available keys: %s",
            outcome,
            summary_path,
            ", ".join(sorted(results)) if results else "<none>",
        )
        return None
    return entry


def _select_row(df: pd.DataFrame, spec: SpecKey) -> pd.Series | None:
    mask = (df["report"] == spec.report) & (df["outcome"] == spec.outcome)
    if spec.tag is not None:
        mask &= df["tag"].fillna("") == spec.tag
    matches = df.loc[mask]
    if matches.empty:
        message = f"No compiled result found for report='{spec.report}', outcome='{spec.outcome}'"
        if spec.tag is not None:
            message += f", tag='{spec.tag}'"
        if spec.required:
            raise click.ClickException(message)
        LOGGER.warning("[skip] %s", message)
        return None
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple rows matched report='%s'. Using the first entry (tags=%s).",
            spec.report,
            matches["tag"].tolist(),
        )
    return matches.iloc[0]


def _write_latex_table(
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
    *,
    caption: str,
    caption_note: str | None = None,
    label: str,
    column_format: str,
    output_path: Path,
) -> None:
    table_df = pd.DataFrame(rows, columns=columns)
    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    if "\\begin{table}" in latex_str and "\\centering" not in latex_str:
        latex_str = latex_str.replace("\\begin{table}", "\\begin{table}\n\\centering\n", 1)
    if caption_note:
        caption_note = caption_note.strip()
        if caption_note:
            replacement = "\\caption*{" + caption_note + "}\n\\end{table}"
            latex_str = latex_str.replace("\\end{table}", replacement, 1)
    output_path.write_text(latex_str, encoding="utf-8")
    LOGGER.info("Wrote LaTeX table → %s", output_path)


def _save_figure(fig: mpl.figure.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    LOGGER.info("Saved figure → %s (.png/.pdf)", base_path)


def build_main_ardl_table(df: pd.DataFrame) -> Path:
    specs = [
        SpecKey(report="tv_density_elag1", label=r"$q=1$"),
        SpecKey(report="baseline_tv_density", label=r"$q=3$"),
        SpecKey(report="tv_density_elag7", label=r"$q=7$"),
    ]
    rows: List[dict[str, str]] = []
    for spec in specs:
        row = _select_row(df, spec)
        if row is None:
            continue
        rows.append(
            {
                "Spec": spec.label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(row.get("beta_sum")),
                "HAC s.e.": _format_decimal(row.get("se_sum")),
                r"$p_{\text{HAC}}$": _format_p(row.get("p_sum_param")),
                r"$n$": _format_integer(row.get("nobs")),
                r"$R^2$": _format_decimal(row.get("r2")),
            }
        )

    # Lead placebo row
    lead_spec = SpecKey(report="tv_density_leads3", label=r"Leads sum ($L=3$)")
    lead_row = _select_row(df, lead_spec)
    if lead_row is not None:
        rows.append(
            {
                "Spec": lead_spec.label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(
                    lead_row.get("beta_leads_sum")
                ),
                "HAC s.e.": _format_decimal(lead_row.get("se_leads_sum")),
                r"$p_{\text{HAC}}$": _format_p(lead_row.get("p_leads_param")),
                r"$n$": _format_integer(lead_row.get("nobs")),
                r"$R^2$": _format_decimal(lead_row.get("r2")),
            }
        )

    output_path = _tables_dir() / "T1_main_ardl.tex"
    _write_latex_table(
        rows,
        [
            "Spec",
            r"$\widehat{\beta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            r"$n$",
            r"$R^2$",
        ],
        caption="ARDL short-run effects of Epstein exposure on novelty.",
        caption_note=(
            "Entries are sums of exposure-lag coefficients with "
            "heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors. "
            "The final row reports the sum of exposure leads as a timing placebo."
        ),
        label="tab:main_ardl",
        column_format="lrrrrr",
        output_path=output_path,
    )
    return output_path


def build_full_spec_table(df: pd.DataFrame) -> Path | None:
    # Appendix table with additional diagnostics for the main ARDL specs

    specs = [
        SpecKey(report="tv_density_elag1", label=r"$q=1$"),
        SpecKey(report="baseline_tv_density", label=r"$q=3$"),
        SpecKey(report="tv_density_elag7", label=r"$q=7$"),
    ]

    rows: list[dict[str, str]] = []
    for spec in specs:
        base_row = _select_row(df, spec)
        if base_row is None:
            continue
        rows.append(
            {
                "Spec": spec.label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(base_row.get("beta_sum")),
                "HAC s.e.": _format_decimal(base_row.get("se_sum")),
                r"$p_{\text{HAC}}$": _format_p(base_row.get("p_sum_param")),
                "Sample": _format_sample(base_row.get("start"), base_row.get("end")),
                r"$n$": _format_integer(base_row.get("nobs")),
                r"$R^2$": _format_decimal(base_row.get("r2")),
            }
        )

    if not rows:
        LOGGER.warning("No rows available for the ARDL spec grid; skipping Appendix table A1.")
        return None

    output_path = _tables_dir() / "A1_full_spec_grid.tex"
    _write_latex_table(
        rows,
        [
            "Spec",
            r"$\widehat{\beta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            "Sample",
            r"$n$",
            r"$R^2$",
        ],
        caption="ARDL specification grid for the novelty outcome.",
        caption_note=(
            "Columns report the short-run exposure effect alongside HAC inference, sample window, "
            "and fit statistics."
        ),
        label="tab:appendix_spec_grid",
        column_format="lrrrlrr",
        output_path=output_path,
    )
    return output_path


def build_mmd2_table(df: pd.DataFrame) -> Path | None:
    specs = [
        SpecKey(report="tv_density_mmd2_elag1", label=r"$q=1$", required=False),
        SpecKey(report="tv_density_mmd2_baseline", label=r"$q=3$", required=False),
        SpecKey(report="tv_density_mmd2_elag7", label=r"$q=7$", required=False),
    ]
    rows: List[dict[str, str]] = []
    for spec in specs:
        row = _select_row(df, spec)
        if row is None:
            continue
        rows.append(
            {
                "Spec": spec.label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(row.get("beta_sum")),
                "HAC s.e.": _format_decimal(row.get("se_sum")),
                r"$p_{\text{HAC}}$": _format_p(row.get("p_sum_param")),
                r"$n$": _format_integer(row.get("nobs")),
                r"$R^2$": _format_decimal(row.get("r2")),
            }
        )

    if not rows:
        LOGGER.warning(
            "No MMD^2 robustness rows were found in the compiled results; skipping table."
        )
        return None

    output_path = _tables_dir() / "A8_mmd2_ardl.tex"
    _write_latex_table(
        rows,
        [
            "Spec",
            r"$\widehat{\beta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            r"$n$",
            r"$R^2$",
        ],
        caption="Alternative novelty measure (MMD$^2$) using Fox exposure.",
        caption_note="Entries mirror Table~\\ref{tab:main_ardl}.",
        label="tab:mmd2_ardl",
        column_format="lrrrrr",
        output_path=output_path,
    )
    return output_path


def build_falsification_table(df: pd.DataFrame) -> Path | None:
    specs = [
        SpecKey(report="falsification_taylorswift", label="Taylor Swift", required=False),
        SpecKey(report="falsification_ncaabasketball", label="NCAA basketball", required=False),
    ]
    rows: List[dict[str, str]] = []
    for spec in specs:
        row = _select_row(df, spec)
        if row is None:
            continue
        rows.append(
            {
                "Keyword": spec.label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(row.get("beta_sum")),
                "HAC s.e.": _format_decimal(row.get("se_sum")),
                r"$p_{\text{HAC}}$": _format_p(row.get("p_sum_param")),
                r"$n$": _format_integer(row.get("nobs")),
            }
        )

    if not rows:
        LOGGER.warning(
            "No falsification ARDL results located; skipping Appendix falsification table."
        )
        return None

    output_path = _tables_dir() / "A7_falsification.tex"
    _write_latex_table(
        rows,
        [
            "Keyword",
            r"$\widehat{\beta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            r"$n$",
        ],
        caption="Falsification checks replacing Epstein exposure with alternative transcript keywords.",
        caption_note=(
            "Effects are expected to be null if the novelty response is specific to the "
            "Epstein coverage spike."
        ),
        label="tab:falsification_tv",
        column_format="lrrrr",
        output_path=output_path,
    )
    return output_path


def build_leads_placebo_table(df: pd.DataFrame) -> Path | None:
    specs = [
        SpecKey(report="tv_density_leads3", label="Fox density ($q=3$)", required=False),
        SpecKey(report="tv_density_mmd2_leads3", label="MMD$^2$ novelty ($q=3$)", required=False),
    ]

    rows: list[dict[str, str]] = []
    for spec in specs:
        row = _select_row(df, spec)
        if row is None:
            continue
        if pd.isna(row.get("beta_leads_sum")):
            continue
        rows.append(
            {
                "Spec": spec.label,
                r"$\widehat{\delta}_{\text{sum}}$": _format_decimal(row.get("beta_leads_sum")),
                "HAC s.e.": _format_decimal(row.get("se_leads_sum")),
                r"$p_{\text{HAC}}$": _format_p(row.get("p_leads_param")),
                "Sample": _format_sample(row.get("start"), row.get("end")),
                r"$n$": _format_integer(row.get("nobs")),
                r"$R^2$": _format_decimal(row.get("r2")),
            }
        )

    if not rows:
        LOGGER.warning("Lead placebo specifications were not found; skipping Appendix table A2.")
        return None

    output_path = _tables_dir() / "A2_leads_placebo.tex"
    _write_latex_table(
        rows,
        [
            "Spec",
            r"$\widehat{\delta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            "Sample",
            r"$n$",
            r"$R^2$",
        ],
        caption="Timing placebo tests using leads of the exposure variable.",
        caption_note="Estimates report the sum of lead coefficients along with HAC inference and model diagnostics.",
        label="tab:appendix_leads",
        column_format="lrrrlrr",
        output_path=output_path,
    )
    return output_path


def build_per_lag_plot(df: pd.DataFrame, reports_dir: Path) -> Path:
    spec = SpecKey(report="baseline_tv_density", label="Baseline per-lag")
    summary_row = _select_row(df, spec)
    if summary_row is None:
        raise click.ClickException(
            "Baseline ARDL results not found; cannot build per-lag plot."
        )
    detail_path = reports_dir / spec.report / "ardl_N_t_z_E_lags_detail.csv"
    if not detail_path.exists():
        raise click.ClickException(
            f"Lag detail CSV missing: {detail_path}. Re-run replication_ardl.sh."
        )
    detail_df = pd.read_csv(detail_path)
    if "lag_index" not in detail_df.columns:
        raise click.ClickException(
            f"Per-lag detail CSV is missing the lag_index column: {detail_path}."
        )
    lag_df = detail_df[detail_df["lag_index"].notna()].copy()
    # Sum and leads rows have NaN lag_index; filter them out for plotting.
    lag_df.sort_values("lag_index", inplace=True)
    if lag_df.empty:
        raise click.ClickException(
            f"No per-lag coefficients found in {detail_path}."
        )

    x = lag_df["lag_index"].to_numpy()
    betas = lag_df["coef"].to_numpy()
    ci = 1.96 * lag_df["se_hac"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.scatter(x, betas, color="#1f78b4", s=50, zorder=3)
    ax.errorbar(
        x,
        betas,
        yerr=ci,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.1,
        capsize=4,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xlabel(r"Lag $j$")
    ax.set_ylabel(r"Coefficient on $E_{t-j}$ (sd)")
    ax.set_title("ARDL per-lag coefficients (Fox exposure, $q=3$)")

    beta_sum = summary_row.get("beta_sum")
    if beta_sum is not None and np.isfinite(beta_sum):
        text = rf"$\widehat{{\beta}}_{{\text{{sum}}}} = {beta_sum:.3f}$"
        ax.text(
            0.98,
            0.05,
            text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#f5f5f5",
                "edgecolor": "#cccccc",
            },
        )

    ax.set_ylim(
        min(0, betas.min() - ci.max() * 1.1),
        max(betas.max() + ci.max() * 1.1, 0),
    )
    fig.tight_layout()

    output_base = _figures_dir() / "F2_per_lag_ardl"
    _save_figure(fig, output_base)
    plt.close(fig)
    return output_base.with_suffix(".pdf")


def build_lp_window_table(reports_dir: Path) -> Path | None:
    entry = _load_lp_result(reports_dir)
    if entry is None:
        return None

    windows: Iterable[dict] = entry.get("windows", [])
    summary = entry.get("summary", {})
    if not windows and summary:
        windows = summary.get("windows", [])

    if not windows:
        LOGGER.warning("No LP window aggregates available in the summary; skipping table A5.")
        return None

    target_windows = [(-3, -1), (0, 1), (0, 3), (0, 7), (0, 14)]
    rows: list[dict[str, str]] = []
    for start, end in target_windows:
        match = next(
            (
                w
                for w in windows
                if int(w.get("start", 0)) == start and int(w.get("end", 0)) == end
            ),
            None,
        )
        if match is None:
            continue
        label = f"{start}–{end}"
        rows.append(
            {
                "Window": label,
                "Estimate": _format_decimal(match.get("coef")),
                "HAC s.e.": _format_decimal(match.get("se")),
                r"$t$": _format_decimal(match.get("t")),
                r"$p_{\text{HAC}}$": _format_p(match.get("p")),
                "Uniform band excl. 0?": _uniform_excludes_zero(
                    match.get("uniform_lower"), match.get("uniform_upper")
                ),
                r"$n$": _format_integer(match.get("nobs")),
            }
        )

    if not rows:
        LOGGER.warning("Target LP windows not found; skipping table A5.")
        return None

    output_path = _tables_dir() / "A5_lp_windows.tex"
    _write_latex_table(
        rows,
        [
            "Window",
            "Estimate",
            "HAC s.e.",
            r"$t$",
            r"$p_{\text{HAC}}$",
            "Uniform band excl. 0?",
            r"$n$",
        ],
        caption=r"Local-projection cumulative windows for the novelty response.",
        caption_note=(
            r"Rows list the estimated impulse responses over selected horizons together "
            r"with HAC inference and whether the 95\% uniform confidence band excludes "
            r"zero."
        ),
        label="tab:appendix_lp_windows",
        column_format="lrrrrlr",
        output_path=output_path,
    )
    return output_path


def build_pretrend_tests_table(reports_dir: Path) -> Path | None:
    entry = _load_lp_result(reports_dir)
    if entry is None:
        return None

    summary = entry.get("summary", {})
    pretrend = summary.get("pretrend_tests") or entry.get("pretrend_tests")
    if not pretrend:
        LOGGER.warning("Pre-trend diagnostics missing from LP summary; skipping table A10.")
        return None

    rows: list[dict[str, str]] = []

    if pretrend.get("wald_p") is not None:
        rows.append(
            {
                "Test": "Wald $H_0$: no pre-trend",
                "Statistic": _format_decimal(pretrend.get("wald_stat")),
                r"$\text{df}$": _format_integer(pretrend.get("wald_df")),
                "p-value": _format_p(pretrend.get("wald_p")),
                "Notes": "",
            }
        )

    if not rows:
        LOGGER.warning("No usable pre-trend diagnostics found; skipping table A10.")
        return None

    output_path = _tables_dir() / "A10_pretrend_tests.tex"
    _write_latex_table(
        rows,
        ["Test", "Statistic", r"$\text{df}$", "p-value", "Notes"],
        caption=r"Joint pre-trend diagnostics accompanying the local-projection event study.",
        caption_note=r"Rows report Wald tests over $h \in [-5,-1]$.",
        label="tab:appendix_pretrend",
        column_format="lrrrl",
        output_path=output_path,
    )
    return output_path


def build_robustness_forest(df: pd.DataFrame) -> Path | None:
    specs = [
        SpecKey(report="tv_density_elag1", label=r"$N_t$: Fox density ($q=1$)"),
        SpecKey(report="baseline_tv_density", label=r"$N_t$: Fox density ($q=3$)"),
        SpecKey(report="tv_density_elag7", label=r"$N_t$: Fox density ($q=7$)"),
        SpecKey(
            report="tv_density_mmd2_elag1",
            label=r"MMD$^2$: Fox density ($q=1$)",
            required=False,
        ),
        SpecKey(
            report="tv_density_mmd2_elag7",
            label=r"MMD$^2$: Fox density ($q=7$)",
            required=False,
        ),
        SpecKey(
            report="gtrends_epstein_elag3",
            label=r"$N_t$: Google Trends Epstein ($q=3$)",
            required=False,
        ),
    ]

    records: List[dict[str, float | str]] = []
    for spec in specs:
        row = _select_row(df, spec)
        if row is None:
            continue
        records.append(
            {
                "label": spec.label,
                "beta": float(row.get("beta_sum")),
                "se": float(row.get("se_sum")),
                "p_hac": row.get("p_sum_param"),
            }
        )

    if not records:
        LOGGER.warning("No robustness specifications found; skipping forest plot.")
        return None

    labels = [record["label"] for record in records][::-1]
    betas = np.array([record["beta"] for record in records][::-1])
    ses = np.array([record["se"] for record in records][::-1])
    p_hac = [record.get("p_hac") for record in records][::-1]
    y_positions = np.arange(len(labels))

    fig_height = max(2.2, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(6.2, fig_height))
    ci95 = 1.96 * ses
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.errorbar(
        betas,
        y_positions,
        xerr=ci95,
        fmt="o",
        color="#1f78b4",
        ecolor="#1f78b4",
        elinewidth=1.1,
        capsize=4,
    )

    for beta, y, hac_p in zip(betas, y_positions, p_hac):
        if hac_p is not None and np.isfinite(hac_p) and hac_p < 0.05:
            ax.plot(beta, y, marker="|", color="#2ca02c", markersize=14, markeredgewidth=2.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Short-run effect on novelty (sd)")
    ax.set_title("") # "Robustness of short-run effect across exposure and novelty measures")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="#1f78b4", linestyle="", label="Estimate ±95% HAC CI"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", 
          bbox_to_anchor=(1.02, 0)) # loc="upper right" , bbox_to_anchor=(1, 0.1))
    fig.tight_layout()

    output_base = _figures_dir() / "F3_robustness_forest"
    _save_figure(fig, output_base)
    plt.close(fig)
    return output_base.with_suffix(".pdf")


def build_exposure_robustness_table(summary_csv: Path | None = None) -> Path | None:
    if summary_csv is None:
        summary_csv = (
            paths.processed_data_dir(ensure=False)
            / "runs"
            / "ardl_robustness"
            / "robustness_summary.csv"
        )

    if not summary_csv.exists():
        LOGGER.warning(
            "Exposure robustness summary not found at %s; skipping Appendix table A9.",
            summary_csv,
        )
        return None

    data = pd.read_csv(summary_csv)
    if data.empty:
        LOGGER.warning("Exposure robustness summary CSV is empty; skipping table A9.")
        return None

    data = data[data["outcome"].fillna("").str.strip() == "N_t_z"]

    label_order = [
        ("tv_epstein_daily_fox_epstein_density", "Fox News density"),
        ("tv_epstein_daily_fox_epstein_hits", "Fox News mentions"),
        ("tv_epstein_daily_msnbc_epstein_density", "MSNBC density"),
        ("tv_epstein_daily_cnn_epstein_density", "CNN density"),
        ("tv_epstein_daily_all3_epstein_mean", "Cable mean (Fox+MSNBC+CNN)"),
        ("gtrends_epstein", "Google Trends: Epstein"),
    ]

    rows: list[dict[str, str]] = []
    for tag, label in label_order:
        match = data[data["tag"] == tag]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "Exposure": label,
                r"$\widehat{\beta}_{\text{sum}}$": _format_decimal(row.get("beta_sum")),
                "HAC s.e.": _format_decimal(row.get("se_sum")),
                r"$p_{\text{HAC}}$": _format_p(row.get("p_sum_param")),
                "Sample": _format_sample(row.get("start"), row.get("end")),
                r"$n$": _format_integer(row.get("nobs")),
                r"$R^2$": _format_decimal(row.get("r2")),
            }
        )

    if not rows:
        LOGGER.warning(
            "None of the expected exposure robustness rows were found in %s; skipping table A9.",
            summary_csv,
        )
        return None

    output_path = _tables_dir() / "A9_exposure_robustness.tex"
    _write_latex_table(
        rows,
        [
            "Exposure",
            r"$\widehat{\beta}_{\text{sum}}$",
            "HAC s.e.",
            r"$p_{\text{HAC}}$",
            "Sample",
            r"$n$",
            r"$R^2$",
        ],
        caption="Robustness of the short-run novelty response across alternative exposure measures.",
        caption_note=(
            "All specifications use the novelty outcome $N_t$ with $q=3$ exposure lags."
        ),
        label="tab:appendix_exposure",
        column_format="lrrrlrr",
        output_path=output_path,
    )
    return output_path


@click.command()
@click.option(
    "--compiled-results",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to ardl_results_compiled.csv (defaults to out/tables/ardl_results_compiled.csv).",
)
@click.option(
    "--reports-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing per-run ARDL reports (defaults to out/reports).",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    help="Enable debug logging.",
)
def main(compiled_results: Path | None, reports_dir: Path | None, verbose: bool) -> None:
    """Build the manuscript tables and figures from ARDL replication outputs."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")
    compiled_df = _read_compiled_results(compiled_results)
    reports_path = _ensure_reports_dir(reports_dir)

    build_main_ardl_table(compiled_df)
    build_full_spec_table(compiled_df)
    build_mmd2_table(compiled_df)
    build_falsification_table(compiled_df)
    build_leads_placebo_table(compiled_df)
    build_per_lag_plot(compiled_df, reports_path)
    build_lp_window_table(reports_path)
    build_pretrend_tests_table(reports_path)
    build_robustness_forest(compiled_df)
    build_exposure_robustness_table()


if __name__ == "__main__":
    main()
