#!/usr/bin/env python3
"""
General utility - sum stats for any dataset


python src/analyze/summary_stats.py --data data_processed/tv_transcripts/daily_fox_epstein.csv --format latex--tag fox_epstein

python src/analyze/summary_stats.py --data data_processed/novelty_energy.parquet --format latex --tag energy


"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import click
import pandas as pd

# Allow running via ``python src/...``
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths, run_metadata


_DEFAULT_CANDIDATES: tuple[str, ...] = (
    "N_t_z",
    "N_t",
    "E_t_z",
    "E_t",
    "V_t_z",
    "V_t",
    "n_posts",
    "novelty_posts_z",
    "daily_fox_epstein_density",
    "daily_all3_epstein_mean",
    "daily_cnn_epstein_density",
    "daily_msnbc_epstein_density",
    "daily_google_epstein_hits",
)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "summary"


def _read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    date_cols = [c for c in df.columns if c.lower() in {"day", "date"}]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = {}
    for col in columns:
        series = pd.to_numeric(df.get(col), errors="coerce")
        if series is not None:
            result[col] = series
    return pd.DataFrame(result)


def _select_columns(df: pd.DataFrame, requested: tuple[str, ...]) -> list[str]:
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise click.UsageError(
                "Requested columns not found: " + ", ".join(sorted(missing))
            )
        return list(requested)

    available = [col for col in _DEFAULT_CANDIDATES if col in df.columns]
    if available:
        return available

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise click.UsageError("No numeric columns detected in the dataset.")
    return numeric_cols


def _summaries(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        missing = series.isna().sum()
        missing_pct = series.isna().mean()
        valid = series.dropna()
        row = {
            "variable": col,
            "count": int(valid.count()),
            "mean": float(valid.mean()) if not valid.empty else float("nan"),
            "std": float(valid.std(ddof=1)) if valid.count() > 1 else float("nan"),
            "min": float(valid.min()) if not valid.empty else float("nan"),
            "p25": float(valid.quantile(0.25)) if not valid.empty else float("nan"),
            "median": float(valid.median()) if not valid.empty else float("nan"),
            "p75": float(valid.quantile(0.75)) if not valid.empty else float("nan"),
            "max": float(valid.max()) if not valid.empty else float("nan"),
            "missing": int(missing),
            "missing_pct": float(missing_pct),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Preserve integer counts before rounding the floating-point statistics.
    if not summary.empty:
        summary["count"] = summary["count"].astype(int)
        summary["missing"] = summary["missing"].astype(int)

        float_cols = [
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "missing_pct",
        ]
        summary[float_cols] = summary[float_cols].round(digits)
    return summary


def _write_output(summary: pd.DataFrame, slug: str, formats: tuple[str, ...], digits: int) -> dict[str, str]:
    tables_dir = paths.tables_dir(ensure=True)
    outputs: dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower()
        target = tables_dir / f"summary_stats_{slug}.{fmt}"
        if fmt == "csv":
            summary.to_csv(target, index=False)
        elif fmt == "json":
            summary.to_json(target, orient="records", indent=2)
        elif fmt == "latex":
            # Column-specific formatting: keep integers as integers;
            # show non-integer floats with the requested precision.
            def _fmt_value(x: float) -> str:
                if pd.isna(x):
                    return ""
                try:
                    xf = float(x)
                    if xf.is_integer():
                        return f"{int(round(xf))}"
                    return f"{xf:.{digits}f}"
                except Exception:
                    return str(x)

            float_cols = [
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
                "missing_pct",
            ]
            formatters = {c: _fmt_value for c in float_cols if c in summary.columns}
            # For LaTeX - replace underscores in the variable names with spaces.
            df_display = summary.copy()
            if "variable" in df_display.columns:
                df_display["variable"] = (
                    df_display["variable"].astype(str).str.replace("_", " ", regex=False)
                )
            target.write_text(
                df_display.to_latex(index=False, formatters=formatters),
                encoding="utf-8",
            )
        elif fmt in {"md", "markdown"}:
            try:
                content = summary.to_markdown(index=False)
            except ImportError:
                content = summary.to_string(index=False)
            target.write_text(content, encoding="utf-8")
        else:
            raise click.UsageError(f"Unsupported output format: {fmt}")
        outputs[fmt] = str(target)
    return outputs


@click.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to the dataset (CSV or Parquet).")
@click.option("--column", "columns", type=str, multiple=True,
              help="Column(s) to include. Defaults to a curated set of key variables.")
@click.option("--tag", type=str, default="", show_default=False,
              help="Optional tag for the output filename.")
@click.option("--format", "formats", type=click.Choice(["csv", "json", "latex", "markdown", "md"], case_sensitive=False),
              multiple=True, default=("csv", "latex"), show_default=True,
              help="One or more output formats.")
@click.option("--digits", default=3, show_default=True, type=click.IntRange(0, 10),
              help="Number of decimal places to round summary statistics.")
def main(data_path: Path, columns: tuple[str, ...], tag: str, formats: tuple[str, ...], digits: int) -> None:
    df = _read_data(data_path)
    selected = _select_columns(df, columns)
    numeric_df = _coerce_numeric(df, selected)
    summary = _summaries(numeric_df, digits)

    slug_base = data_path.stem or "data"
    slug_parts = [slug_base]
    if tag:
        slug_parts.append(tag)
    slug = _slugify("_".join(slug_parts))

    outputs = _write_output(summary, slug, formats or ("csv",), digits)

    run_info = run_metadata.record_run(
        components=("summary_stats", slug),
        parameters={
            "columns": selected,
            "digits": digits,
            "formats": list(formats),
        },
        inputs={"data": str(data_path)},
        outputs=outputs,
    )

    click.echo("Summary statistics generated:")
    for fmt, path in outputs.items():
        click.echo(f"  - {fmt}: {path}")
    click.echo(f"Metadata recorded at {run_info.metadata_path}")


if __name__ == "__main__":
    main()
