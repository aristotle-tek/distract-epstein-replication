#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


def _infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    # this is dumb but at least it should work - try common spots
    candidates = [
        Path.cwd(),
        here.parent,  # src/
        here.parent.parent,  # repo root (if script is in src/)
        Path.cwd().parent,  # running from src/: cwd is src/
    ]
    for base in candidates:
        if (base / "data_processed" / "tv_transcripts").exists():
            return base
    # Last resort: walk up looking for "data_processed/tv_transcripts"
    p = here
    for _ in range(6):
        if (p / "data_processed" / "tv_transcripts").exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Could not locate 'data_processed/tv_transcripts'. "
        "Run from the repo root or pass data_root=... explicitly."
    )


def _slugify(term: str) -> str:
    value = term.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "series"


def _resolve_paths(
    base: Path, files: Mapping[str, str | Path]
) -> "OrderedDict[str, Path]":
    resolved: "OrderedDict[str, Path]" = OrderedDict()
    for network, raw in files.items():
        if raw is None:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Missing file for '{network}': {candidate}")
        resolved[network] = candidate
    if not resolved:
        raise ValueError("At least one network CSV must be provided.")
    return resolved


def _load_one(path: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["hits"] = pd.to_numeric(df.get("hits"), errors="coerce")
    df["density_per_1000"] = pd.to_numeric(df.get("density_per_1000"), errors="coerce")
    out = df[["date", "hits", "density_per_1000"]].rename(
        columns={"hits": f"{key}_hits", "density_per_1000": f"{key}_density"}
    )
    return out


def _print_summary(base: Path, paths: Mapping[str, Path], frames: Mapping[str, pd.DataFrame], merged: pd.DataFrame) -> None:
    def _date_range(df: pd.DataFrame, label: str) -> None:
        dmin, dmax, n = df["date"].min(), df["date"].max(), df.shape[0]
        smin = dmin.date() if pd.notna(dmin) else None
        smax = dmax.date() if pd.notna(dmax) else None
        print(f"{label:28s} rows={n:5d}  start={smin}  end={smax}")

    print("Resolved base directory:", base.resolve())
    print("CSV paths:")
    for network, path in paths.items():
        print(f"  {network:>8s}: {path}")

    print("\nPer-file date ranges (before merge):")
    for label, frame in frames.items():
        _date_range(frame, label)

    print("\nFinal merged date range:")
    _date_range(merged, "merged_all")

    miss_report = {label: merged[f"{label}_hits"].isna().sum() for label in frames}
    print("\nMissing days per network (NaNs in *_hits after outer-join):")
    for label, count in miss_report.items():
        print(f"  {label:24s}: {count}")


def merge_tv_transcripts(
    *,
    files: Mapping[str, str | Path],
    label: str,
    slug: str | None = None,
    data_root: str | Path | None = None,
    output_path: str | Path | None = None,
    print_summary: bool = True,
) -> pd.DataFrame:
    base = Path(data_root) if data_root is not None else _infer_repo_root()
    resolved_paths = _resolve_paths(base, files)
    dataset_slug = slug or _slugify(label)

    frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for network, path in resolved_paths.items():
        key = f"daily_{network}_{dataset_slug}"
        frames[key] = _load_one(path, key)

    merged = None
    for frame in frames.values():
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")

    if merged is None:
        raise RuntimeError("Merging failed; no frames were loaded.")

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    hit_cols = [c for c in merged.columns if c.endswith("_hits")]
    dens_cols = [c for c in merged.columns if c.endswith("_density")]

    n_networks = len(frames)
    merged[f"daily_all{n_networks}_{dataset_slug}_hits"] = merged[hit_cols].sum(axis=1, skipna=True)
    merged[f"daily_all{n_networks}_{dataset_slug}_mean"] = merged[dens_cols].mean(axis=1, skipna=True)

    if print_summary:
        _print_summary(base, resolved_paths, frames, merged)

    if output_path is not None:
        out_candidate = Path(output_path)
        if not out_candidate.is_absolute():
            out_candidate = (base / out_candidate).resolve()
        else:
            out_candidate = out_candidate.resolve()
        out_candidate.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_candidate, index=False)
        print(f"\nSaved merged CSV to: {out_candidate}")

    return merged


def load_merge_epstein_tv(
    data_root: str | Path | None = None,
    fox_rel="data_processed/tv_transcripts/daily_fox_epstein.csv",
    cnn_rel="data_processed/tv_transcripts/daily_cnn_epstein.csv",
    msnbc_rel="data_processed/tv_transcripts/daily_msnbc_epstein.csv",
):
    return merge_tv_transcripts(
        files={
            "fox": fox_rel,
            "cnn": cnn_rel,
            "msnbc": msnbc_rel,
        },
        label="Epstein",
        slug="epstein",
        data_root=data_root,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for TV transcript merges.")
    subparsers = parser.add_subparsers(dest="command")

    merge = subparsers.add_parser(
        "merge-tv",
        help="Merge per-network transcript CSVs into a combined dataset.",
    )
    merge.add_argument("--label", required=True, help="Human-readable label for the dataset.")
    merge.add_argument(
        "--slug",
        help="Slug used in output column names (defaults to a slugified label).",
    )
    merge.add_argument("--data-root", dest="data_root", help="Optional repository root override.")
    merge.add_argument("--out", dest="output", help="Optional output CSV path.")
    merge.add_argument("--fox", help="Fox News daily CSV path.")
    merge.add_argument("--cnn", help="CNN daily CSV path.")
    merge.add_argument("--msnbc", help="MSNBC daily CSV path.")
    merge.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress informational summary output.",
    )

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "merge-tv":
        files: "OrderedDict[str, str | Path]" = OrderedDict()
        for key, value in (("fox", args.fox), ("cnn", args.cnn), ("msnbc", args.msnbc)):
            if value is not None:
                files[key] = value
        try:
            merge_tv_transcripts(
                files=files,
                label=args.label,
                slug=args.slug,
                data_root=args.data_root,
                output_path=args.output,
                print_summary=not args.no_summary,
            )
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
