#!/usr/bin/env python3

# Merge post embedding parquet parts into a single file (split to keep under github limits without gfs)

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import click
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INPUT_PATTERN = "data_processed/post_embeddings.part*.parquet"
DEFAULT_OUTPUT = Path("data_processed/post_embeddings.parquet")
PART_REGEX = re.compile(r"\.part(\d+)of(\d+)\.parquet$")


def _resolve_parts(pattern: str) -> list[Path]:
    paths = list(Path().glob(pattern))
    if not paths:
        raise click.UsageError(f"No files matched pattern: {pattern}")

    def sort_key(path: Path):
        match = PART_REGEX.search(path.name)
        if match:
            part_idx = int(match.group(1))
            total_parts = int(match.group(2))
            return (path.parent, total_parts, part_idx)
        return (path.parent, float("inf"), path.name)

    return sorted(paths, key=sort_key)


def _load_manifest(manifest_path: Path) -> list[dict] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text()).get("parts", [])
    except Exception as exc:
        raise click.ClickException(f"Failed to read manifest {manifest_path}: {exc}") from exc


def _validate_against_manifest(parts: Iterable[Path], manifest_entries: list[dict]) -> None:
    if manifest_entries is None:
        return

    part_names = [p.name for p in parts]
    manifest_names = [entry.get("file") for entry in manifest_entries]
    if part_names != manifest_names:
        raise click.ClickException(
            "Parts on disk do not match manifest order.\n"
            f"Manifest: {manifest_names}\n"
            f"Found:    {part_names}"
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-pattern",
    "input_pattern",
    default=DEFAULT_INPUT_PATTERN,
    show_default=True,
    help="Glob pattern for embedding part Parquet files.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Destination Parquet path for merged embeddings.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
@click.option(
    "--skip-manifest",
    is_flag=True,
    help="Skip validating against the embedding manifest.",
)
def main(input_pattern: str, output_path: Path, force: bool, skip_manifest: bool) -> None:

    parts = _resolve_parts(input_pattern)

    if output_path.exists() and not force:
        raise click.ClickException(
            f"Output already exists: {output_path}. Use --force to overwrite."
        )

    manifest_entries = None
    manifest_path = output_path.with_suffix(".parts.json")
    if not skip_manifest:
        manifest_entries = _load_manifest(manifest_path)
        if manifest_entries is not None:
            _validate_against_manifest(parts, manifest_entries)
            click.echo(
                f"Validated {len(parts)} parts against manifest {manifest_path}"
            )
        else:
            click.echo("No manifest found; proceeding without validation.")
    else:
        click.echo("Skipping manifest validation as requested.")

    tables: list[pa.Table] = []
    for path in parts:
        click.echo(f"Reading {path} ...")
        tables.append(pq.read_table(path))

    if not tables:
        raise click.ClickException("No tables were read from matched parts.")

    merged = pa.concat_tables(tables, promote=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, output_path)
    click.echo(f"Wrote merged embeddings → {output_path}")


if __name__ == "__main__":
    main()
