# src/rl_fzerox/core/manager/artifacts/filesystem.py
"""Persisted filesystem mutations owned by the manager registry."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FilesystemOperationKind = Literal["delete_tree", "move_tree"]


@dataclass(frozen=True, slots=True)
class FilesystemOperation:
    """One persisted filesystem mutation that must survive manager restarts."""

    id: int
    kind: FilesystemOperationKind
    source_path: Path
    target_path: Path | None
    created_at: str


def filesystem_operation_from_values(
    *,
    operation_id: int,
    kind: object,
    source_path: object,
    target_path: object,
    created_at: object,
) -> FilesystemOperation:
    """Decode one persisted filesystem operation."""

    return FilesystemOperation(
        id=operation_id,
        kind=_filesystem_operation_kind(kind),
        source_path=Path(str(source_path)).expanduser().resolve(),
        target_path=_optional_path(target_path),
        created_at=str(created_at),
    )


def apply_filesystem_operation(operation: FilesystemOperation) -> bool:
    """Apply one persisted operation, returning True when it is complete."""

    if operation.kind == "delete_tree":
        return _delete_tree(operation.source_path)
    return _move_tree(
        source_path=operation.source_path,
        target_path=_required_path(operation.target_path),
    )


def _delete_tree(path: Path) -> bool:
    if not path.exists():
        return True
    shutil.rmtree(path)
    return True


def _move_tree(*, source_path: Path, target_path: Path) -> bool:
    if source_path == target_path:
        return True
    if source_path.exists():
        if target_path.exists():
            raise RuntimeError(
                "filesystem move cannot complete because both source and target exist: "
                f"{source_path} -> {target_path}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        return True
    if target_path.exists():
        return True
    raise RuntimeError(
        "filesystem move cannot complete because neither source nor target exists: "
        f"{source_path} -> {target_path}"
    )


def _filesystem_operation_kind(value: object) -> FilesystemOperationKind:
    if value == "delete_tree":
        return "delete_tree"
    if value == "move_tree":
        return "move_tree"
    raise ValueError(f"Unsupported filesystem operation kind: {value!r}")


def _optional_path(value: object) -> Path | None:
    return Path(str(value)).expanduser().resolve() if isinstance(value, str) else None


def _required_path(value: Path | None) -> Path:
    if value is None:
        raise ValueError("filesystem move operation is missing its target path")
    return value
