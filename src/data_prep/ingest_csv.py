"""Utility to clean a truth social csv and output parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.utils import paths


def strip_html(s: str) -> str:
    try:
        return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    except Exception:
        return s


def ingest_csv(csv_path: Path, parquet_out: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="id")
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.strip().ne("")]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["day"] = df["created_at"].dt.floor("D")
    df["content_clean"] = df["content"].map(strip_html)

    paths.ensure_directory(parquet_out.parent)
    df.to_parquet(parquet_out, index=False)
    print(f"Saved {len(df)} rows → {parquet_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Truth Social csv to parquet.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional path to configuration file overriding config.yml")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Input csv (default: paths.truth_archive_csv())")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output parquet (default: paths.truth_posts_parquet())")
    args = parser.parse_args()

    if args.config:
        paths.configure(args.config)

    csv_path = args.csv or paths.truth_archive_csv()
    parquet_out = args.out or paths.truth_posts_parquet()

    ingest_csv(Path(csv_path), Path(parquet_out))


if __name__ == "__main__":
    main()
