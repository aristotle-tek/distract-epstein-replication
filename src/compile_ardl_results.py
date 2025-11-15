#!/usr/bin/env python3


# wrapper to compile ARDL report summaries into a single csv

# python src/compile_ardl_results.py --reports-dir out/reports --output out/tables/ardl_results_compiled.csv


from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable, List

import click
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths


def _iter_summary_paths(base_dir: Path, pattern: str) -> Iterable[Path]:
    for report_dir in sorted(base_dir.iterdir()):
        if not report_dir.is_dir():
            continue
        for summary_path in sorted(report_dir.glob(pattern)):
            if summary_path.is_file():
                yield summary_path


def _derive_tag(summary_filename: str) -> str:
    stem = summary_filename[:-5] if summary_filename.endswith(".json") else summary_filename
    prefix = "ardl_summary"
    if stem == prefix:
        return ""
    if stem.startswith(prefix + "_"):
        return stem[len(prefix) + 1 :]
    return ""


def _ensure_absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (paths.repo_root() / path).resolve()


def _safe_relative_to_repo(path: Path) -> Path:
    resolved = path.resolve()
    repo_root = paths.repo_root()
    try:
        return resolved.relative_to(repo_root)
    except ValueError:
        return resolved


@click.command()
@click.option(
    "--reports-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing report folders (defaults to config paths.reports_dir).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination CSV file (defaults to out/tables/ardl_results_compiled.csv).",
)
@click.option(
    "--pattern",
    default="ardl_summary*.json",
    show_default=True,
    help="Glob pattern for summary files inside each report directory.",
)
@click.option(
    "--quiet/--no-quiet",
    default=False,
    show_default=True,
    help="Suppress stdout table preview.",
)
def main(reports_dir: Path | None, output: Path | None, pattern: str, quiet: bool) -> None:

    if reports_dir is None:
        reports_dir = paths.reports_dir(ensure=False)
    reports_dir = _ensure_absolute(reports_dir)
    if not reports_dir.exists():
        raise click.ClickException(f"Reports directory does not exist: {reports_dir}")

    summary_paths: List[Path] = list(_iter_summary_paths(reports_dir, pattern))
    if not summary_paths:
        raise click.ClickException(
            f"No summary files matching '{pattern}' were found under {reports_dir}"
        )

    records = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload.get("_meta", {})
        report_name = summary_path.parent.name
        tag = _derive_tag(summary_path.name)
        for outcome, values in payload.items():
            if outcome == "_meta":
                continue
            record = {
                "report": report_name,
                "summary_file": summary_path.name,
                "summary_path": str(
                    _safe_relative_to_repo(summary_path)
                ),
                "tag": tag,
                "outcome": outcome,
                "beta_sum": values.get("beta_sum"),
                "se_sum": values.get("se_sum"),
                "p_sum_param": values.get("p_sum_param"),
                "nobs": values.get("nobs"),
                "r2": values.get("r2"),
                "start": values.get("start"),
                "end": values.get("end"),
                "beta_leads_sum": values.get("beta_leads_sum"),
                "se_leads_sum": values.get("se_leads_sum"),
                "p_leads_param": values.get("p_leads_param"),
                "day_panel": meta.get("day_panel"),
                "exposure_argument": meta.get("exposure_argument"),
                "exposure_base_column": meta.get("exposure_base_column"),
                "exposure_column_used": meta.get("exposure_column_used"),
                "use_e_z": meta.get("use_e_z"),
                "standardized_column_created": meta.get("standardized_column_created"),
            }
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df.sort_values(by=["report", "tag", "outcome"], inplace=True)

    if output is None:
        tables_dir = paths.tables_dir(ensure=True)
        output = tables_dir / "ardl_results_compiled.csv"
    else:
        output = _ensure_absolute(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)
    click.echo(f"Wrote combined results to {output}")

    if not quiet:
        display_cols = [
            "report",
            "tag",
            "outcome",
            "beta_sum",
            "p_sum_param",
        ]
        preview = df[display_cols].copy()
        click.echo("\nKey results (first 20 rows):")
        click.echo(preview.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
