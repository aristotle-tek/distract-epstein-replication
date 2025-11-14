"""
Helpers for organizing run directories and logging metadata.
To avoid confusion & collisions
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import paths

_METADATA_FILENAME = "metadata.json"


def _git_revision() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.repo_root(),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.decode().strip() or None


def _sanitize_component(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Run path components must be non-empty strings")
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-._")
    if not text:
        raise ValueError(f"Run path component '{value}' is empty after sanitization")
    if text in {".", ".."}:
        raise ValueError("Run path components cannot be '.' or '..'")
    return text


def run_directory(*components: str, ensure: bool = True) -> Path:
    sanitized = [_sanitize_component(c) for c in components if c]
    base = paths.runs_dir(ensure=ensure)
    run_dir = base.joinpath(*sanitized)
    if ensure:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def metadata_path(run_dir: Path) -> Path:
    return Path(run_dir) / _METADATA_FILENAME


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value


def build_metadata(
    *,
    components: Sequence[str],
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    notes: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # metadata for a run
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "components": list(components),
    }

    revision = _git_revision()
    if revision:
        payload["git_commit"] = revision

    if parameters:
        payload["parameters"] = _json_ready(parameters)
    if inputs:
        payload["inputs"] = _json_ready(inputs)
    if outputs:
        payload["outputs"] = _json_ready(outputs)
    if notes:
        payload["notes"] = _json_ready(notes)
    if extras:
        payload["extras"] = _json_ready(extras)

    return payload


def write_metadata(run_dir: Path, metadata: Mapping[str, Any]) -> Path:
    # Persist ``metadata`` to ``run_dir/metadata.json`` and return the path.

    target = metadata_path(run_dir)
    target.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True))
    return target


@dataclass(frozen=True)
class RunInfo:
    components: tuple[str, ...]
    run_dir: Path
    metadata_path: Path


def record_run(
    components: Iterable[str],
    *,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    notes: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> RunInfo:
    # reate a run directory, write metadata, and return a :class:`RunInfo`.

    component_list = tuple(_sanitize_component(c) for c in components if c)
    run_dir = run_directory(*component_list, ensure=True)
    metadata = build_metadata(
        components=component_list,
        parameters=parameters,
        inputs=inputs,
        outputs=outputs,
        notes=notes,
        extras=extras,
    )
    meta_path = write_metadata(run_dir, metadata)
    return RunInfo(components=component_list, run_dir=run_dir, metadata_path=meta_path)
