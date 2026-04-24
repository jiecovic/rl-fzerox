# src/rl_fzerox/core/config/track_registry/common.py
from __future__ import annotations

from pathlib import Path


def safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def optional_weight(raw_weight: object, *, label: str) -> float | None:
    if raw_weight is None:
        return None
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
        raise TypeError(f"{label} must be numeric")
    weight = float(raw_weight)
    if weight <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return weight


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def registry_path(*, root: Path, ref: str, label: str, required: bool = True) -> Path | None:
    registry_root = root.resolve()
    path = (registry_root / ref).with_suffix(".yaml").resolve()
    if not path.is_relative_to(registry_root):
        raise ValueError(f"{label} ref escapes registry root: {ref!r}")
    if not path.is_file():
        if not required:
            return None
        raise FileNotFoundError(f"{label} not found: {ref!r}")
    return path


def nested_mapping(value: dict[str, object], *path: str) -> dict[str, object] | None:
    cursor: object = value
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor if isinstance(cursor, dict) else None
