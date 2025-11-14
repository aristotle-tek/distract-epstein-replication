#!/usr/bin/env python3
"""Compute daily novelty measures from Truth Social embeddings.

Two versions:

``energy``
    Energy distance between the distribution of embeddings on day t and
    the embeddings from the preceding ``window`` calendar days. 

``mmd2``
    Squared maximum mean discrepancy (RBF kernel) comparing day t to the
    trailing window.  The kernel bandwidth defaults to the median pairwise
    distance heuristic computed from the combined sample (override with --kernel-scale )


------

gen the energy-distance series:

    python src/novelty.py build \
        --method energy --window 7 \
        --emb-path data_processed/post_embeddings.parquet \
        --out data_processed/novelty_energy.parquet

MMD2 alternative:

    python src/novelty.py build \
        --method mmd2 --window 30 \
        --emb-path data_processed/post_embeddings.parquet \
        --out data_processed/novelty_mmd2.parquet

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist, pdist

# Allow running the script via ``python src/novelty.py``
if __package__ in {None, ""}:
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from src.utils import paths


@dataclass(frozen=True)
class DaySlice:
    day: pd.Timestamp
    embeddings: np.ndarray


def _unit_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return matrix


def _load_embeddings(path: Path) -> list[DaySlice]:
    # Load embeddings grouped by day from the emb parquet

    table = pq.read_table(path, columns=["day", "emb_white"])
    if table.num_rows == 0:
        return []

    # ``day`` is stored as Arrow date32 -> convert to UTC
    days = pd.to_datetime(table.column("day").to_pandas()).dt.tz_localize("UTC")
    emb_col = table.column("emb_white")
    emb_vectors = [np.asarray(vec, dtype=np.float32) for vec in emb_col.to_pylist()]

    df = pd.DataFrame({"day": days, "embedding": emb_vectors})
    df = df.dropna(subset=["day"]).sort_values("day")
    grouped: list[DaySlice] = []
    for day, grp in df.groupby("day"):
        mats = grp["embedding"].to_list()
        if not mats:
            continue
        matrix = np.vstack(mats).astype(np.float32, copy=False)
        grouped.append(DaySlice(day=day, embeddings=_unit_normalize_rows(matrix)))
    return grouped


def _mean_pairwise_distance(values: np.ndarray) -> float:
    n = values.shape[0]
    if n < 2:
        return 0.0
    dists = pdist(values, metric="euclidean")
    if dists.size == 0:
        return 0.0
    total = dists.sum()
    return float((2.0 * total) / (n * (n - 1)))


def _energy_distance(current: np.ndarray, reference: np.ndarray) -> float:
    n, m = current.shape[0], reference.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    cross = cdist(current, reference, metric="euclidean")
    term1 = 2.0 * cross.mean()
    term2 = _mean_pairwise_distance(current)
    term3 = _mean_pairwise_distance(reference)
    value = term1 - term2 - term3
    return float(max(value, 0.0))


def _rbf_self_sum(x: np.ndarray, gamma: float) -> float:
    n = x.shape[0]
    if n < 2:
        return 0.0
    sq = cdist(x, x, metric="sqeuclidean")
    kernel = np.exp(-gamma * sq)
    return float(kernel.sum() - np.trace(kernel))


def _rbf_cross_sum(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    sq = cdist(x, y, metric="sqeuclidean")
    return float(np.exp(-gamma * sq).sum())


def _estimate_gamma(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack([x, y])
    total = combined.shape[0]
    if total < 2:
        return float("nan")
    distances = pdist(combined, metric="euclidean")
    if distances.size == 0:
        return float("nan")
    median = float(np.median(distances))
    if not np.isfinite(median) or median <= 0:
        positive = distances[distances > 0]
        if positive.size == 0:
            return float("nan")
        median = float(np.median(positive))
    if not np.isfinite(median) or median <= 0:
        return float("nan")
    return 1.0 / (2.0 * median * median)


def _mmd2(current: np.ndarray, reference: np.ndarray, *, gamma: float | None = None) -> float:
    n, m = current.shape[0], reference.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    gamma_val = gamma if gamma is not None else _estimate_gamma(current, reference)
    if not np.isfinite(gamma_val) or gamma_val <= 0:
        return float("nan")
    self_current = _rbf_self_sum(current, gamma_val) / (n * (n - 1))
    self_reference = _rbf_self_sum(reference, gamma_val) / (m * (m - 1))
    cross = _rbf_cross_sum(current, reference, gamma_val) / (n * m)
    value = self_current + self_reference - 2.0 * cross
    return float(max(value, 0.0))


def _iter_results(
    slices: Sequence[DaySlice],
    *,
    window: int,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    method: str,
    gamma: float | None,
    min_posts: int,
    min_reference_posts: int,
) -> Iterator[dict[str, float | pd.Timestamp]]:
    trailing: list[DaySlice] = []
    for slice_ in slices:
        day = slice_.day
        if start and day < start:
            if slice_.embeddings.shape[0] >= min_posts:
                trailing.append(slice_)
            continue
        if end and day > end:
            break

        trailing = [s for s in trailing if s.day >= day - pd.Timedelta(days=window)]
        reference_chunks = [s.embeddings for s in trailing if s.day < day]
        reference = np.vstack(reference_chunks) if reference_chunks else np.empty((0, slice_.embeddings.shape[1]), dtype=np.float32)

        current_count = int(slice_.embeddings.shape[0])
        reference_count = int(reference.shape[0])

        value = float("nan")
        if current_count >= min_posts and reference_count >= min_reference_posts:
            if method == "energy":
                value = _energy_distance(slice_.embeddings, reference)
            else:
                value = _mmd2(slice_.embeddings, reference, gamma=gamma)

        if np.isfinite(value):
            value = max(value, 0.0)

        result: dict[str, float | pd.Timestamp] = {
            "day": day,
            "N_t": value,
            "n_posts": current_count,
            "n_ref_posts": reference_count,
            "min_posts_used": min_posts,
            "min_ref_posts_used": min_reference_posts,
        }

        yield result

        if current_count >= min_posts:
            trailing.append(slice_)


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None,
              help="Optional configuration file overriding config.yml.")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    if config_path:
        paths.configure(config_path)
    ctx.obj = {}


@cli.command("build")
@click.option("--emb-path", type=click.Path(path_type=Path), default=None,
              help="Embeddings parquet (default: paths.post_embeddings_parquet()).")
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None,
              help="Output parquet (default: inferred from method).")
@click.option("--method", type=click.Choice(["energy", "mmd2"], case_sensitive=False), default="energy",
              show_default=True, help="Novelty statistic to compute.")
@click.option("--window", type=int, default=None, show_default=False,
              help="Trailing calendar-day window for the reference sample (default: 7 for energy, 30 for mmd2).")
@click.option("--start", type=str, default=None,
              help="Optional inclusive start date (YYYY-MM-DD, UTC).")
@click.option("--end", type=str, default=None,
              help="Optional inclusive end date (YYYY-MM-DD, UTC).")
@click.option("--kernel-scale", type=float, default=None,
              help="Override the RBF kernel scale (σ) for MMD^2. When omitted the median-distance heuristic is used.")
@click.option("--min-posts", type=int, default=3, show_default=True,
              help="Minimum number of posts required on day t to compute novelty.")
@click.option("--min-reference-posts", type=int, default=None, show_default=False,
              help="Minimum number of posts in the trailing window. Defaults to max(2*min_posts, 10).")
def build_command(
    emb_path: Path | None,
    out_path: Path | None,
    method: str,
    window: int | None,
    start: str | None,
    end: str | None,
    kernel_scale: float | None,
    min_posts: int,
    min_reference_posts: int | None,
) -> None:
    # Compute the novelty, out -> parquet."""

    emb_path = emb_path or paths.post_embeddings_parquet()
    if not emb_path.exists():
        raise click.ClickException(f"Embeddings parquet not found: {emb_path}")

    if out_path is None:
        out_path = paths.novelty_mmd2_parquet() if method.lower() == "mmd2" else paths.novelty_energy_parquet()
    else:
        out_path = Path(out_path)

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    slices = _load_embeddings(emb_path)
    if not slices:
        raise click.ClickException("No embeddings available to compute novelty.")

    method_key = method.lower()
    if method_key not in {"energy", "mmd2"}:
        raise click.ClickException(f"Unsupported method: {method}")

    if window is None:
        window_value = 30 if method_key == "mmd2" else 7
    else:
        window_value = window
    if window_value <= 0:
        raise click.ClickException("--window must be a positive integer.")

    gamma = None
    if method_key == "mmd2" and kernel_scale is not None:
        if kernel_scale <= 0:
            raise click.ClickException("--kernel-scale must be positive when provided.")
        gamma = 1.0 / (2.0 * (kernel_scale ** 2))

    min_posts = int(min_posts)
    if min_posts < 2:
        raise click.ClickException("--min-posts must be at least 2.")

    if min_reference_posts is None:
        min_reference_posts_value = max(min_posts * 2, 10)
    else:
        min_reference_posts_value = int(min_reference_posts)
        if min_reference_posts_value < min_posts:
            raise click.ClickException("--min-reference-posts must be ≥ --min-posts.")

    results = list(
        _iter_results(
            slices,
            window=window_value,
            start=start_ts,
            end=end_ts,
            method=method_key,
            gamma=gamma,
            min_posts=min_posts,
            min_reference_posts=min_reference_posts_value,
        )
    )

    if not results:
        raise click.ClickException("No days fell within the requested range; nothing to write.")

    output_df = pd.DataFrame(results)
    paths.ensure_directory(out_path.parent)
    output_df.to_parquet(out_path, index=False)
    click.echo(f"Wrote novelty series → {out_path}")


if __name__ == "__main__":
    try:
        cli()
    except click.ClickException as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
