"""
for loading paths
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yml"

_CURRENT_CONFIG_PATH: Path = DEFAULT_CONFIG_PATH


def configure(config_path: Path | str) -> None:
    # Override the config file used for path resolution
    global _CURRENT_CONFIG_PATH
    _CURRENT_CONFIG_PATH = Path(config_path)
    _load_config.cache_clear()


@lru_cache(maxsize=1)
def _load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _CURRENT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def get_config() -> Dict[str, Any]:
    # Return the parsed congif dictionary
    return _load_config(str(_CURRENT_CONFIG_PATH))


def repo_root() -> Path:
    return REPO_ROOT


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _paths_cfg() -> Dict[str, Any]:
    return get_config().get("paths", {})


def _path_from_config(key: str, *, ensure: bool = False) -> Path:
    raw_value = _paths_cfg().get(key)
    if raw_value is None:
        raise KeyError(f"Missing 'paths.{key}' in configuration")
    candidate = _resolve_path(raw_value)
    if ensure:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def output_root(*, ensure: bool = False) -> Path:
    raw_value = _paths_cfg().get("output_root", "out")
    root = _resolve_path(raw_value)
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
    return root


def output_subdir(name: str, *, ensure: bool = False) -> Path:
    base = output_root(ensure=ensure)
    path = base / name
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("raw_data_dir", ensure=ensure)


def processed_data_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("processed_data_dir", ensure=ensure)


def runs_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("runs_dir", ensure=ensure)


def outputs_dir(*, ensure: bool = False) -> Path:
    return output_root(ensure=ensure)


def reports_dir(*, ensure: bool = False) -> Path:
    return output_subdir("reports", ensure=ensure)


def reports_gtrends_dir(*, ensure: bool = False) -> Path:
    path = reports_dir(ensure=ensure) / "gtrends"
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_plots_dir(*, ensure: bool = False) -> Path:
    return figures_dir(ensure=ensure)


def figures_dir(*, ensure: bool = False) -> Path:
    return output_subdir("figures", ensure=ensure)


def figures_mirror_for_outdir(outdir: Path, *, ensure: bool = True) -> Path:
    # Return the figures directory that mirrors a given output dir
    outdir = Path(outdir)
    figures_root = figures_dir(ensure=ensure)

    relative: Path | None = None
    for resolver in (reports_dir, output_root):
        try:
            base = resolver()
            relative = outdir.relative_to(base)
            break
        except ValueError:
            continue

    if relative is None or str(relative) == ".":
        relative = Path(outdir.name)

    target = figures_root / relative
    if ensure:
        target.mkdir(parents=True, exist_ok=True)
    return target


def tables_dir(*, ensure: bool = False) -> Path:
    return output_subdir("tables", ensure=ensure)


def logs_dir(*, ensure: bool = False) -> Path:
    return output_subdir("logs", ensure=ensure)


def artifacts_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("artifacts_dir", ensure=ensure)


def truth_social_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("truth_social_dir", ensure=ensure)


def transcripts_processed_dir(*, ensure: bool = False) -> Path:
    return _path_from_config("transcripts_processed_dir", ensure=ensure)


def events_csv_path() -> Path:
    events_rel = _path_from_config("events_csv")
    return events_rel


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_fragment() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def report_run_dir(
    report_name: str,
    *,
    timestamp: bool | str = False,
    ensure: bool = True,
) -> Path:
    """Return a per-run directory under ``output_root/reports``.

    report_name:  Name of the report folder. This is appended to ``output_root/reports``.
    timestamp:
        If truthy, append a ``YYYYMMDD-HHMMSS`` suffix (or the provided string)
        to the final path component to avoid collisions.
    ensure: When T (default), create the dir
    """

    rel = Path(report_name.strip()) if report_name else None
    if not rel or rel.is_absolute() or any(part in {"..", ".", ""} for part in rel.parts):
        raise ValueError("report_name must be a relative path without '.', '..', or empty segments")

    base = reports_dir(ensure=ensure)
    if timestamp:
        suffix = timestamp if isinstance(timestamp, str) else _timestamp_fragment()
        rel = rel.with_name(f"{rel.name}_{suffix}")

    target = base / rel
    if ensure:
        target.mkdir(parents=True, exist_ok=True)
    return target


def repo_path(*relative: str) -> Path:
    return REPO_ROOT.joinpath(*relative)


def truth_archive_csv() -> Path:
    return truth_social_dir() / "truth_archive.csv"


def truth_posts_parquet() -> Path:
    return truth_social_dir() / "truth_posts.parquet"


def post_embeddings_parquet() -> Path:
    return processed_data_dir() / "post_embeddings.parquet"


def truth_bucket_labels_dir(timezone: str = "UTC", *, ensure: bool = False) -> Path:
    base = output_subdir("truth_buckets", ensure=ensure) / "labels"
    slug = (
        timezone.strip().lower().replace("/", "__").replace(":", "-").replace(" ", "_")
    ) or "tz"
    slug = re.sub(r"[^a-z0-9_.-]", "_", slug)
    path = base / slug
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def truth_bucket_daily_shares_parquet() -> Path:
    path = processed_data_dir() / "truth_buckets" / "daily_shares.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def novelty_energy_parquet() -> Path:
    return processed_data_dir() / "novelty_energy.parquet"


def novelty_mmd2_parquet() -> Path:
    return processed_data_dir() / "novelty_mmd2.parquet"


def day_panel_parquet() -> Path:
    return processed_data_dir() / "day_panel.parquet"
