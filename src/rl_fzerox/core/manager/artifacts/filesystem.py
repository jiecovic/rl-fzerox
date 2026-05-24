# src/rl_fzerox/core/manager/artifacts/filesystem.py
"""Persisted filesystem mutations owned by the manager registry."""

from __future__ import annotations

import shutil
import sqlite3
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


def queue_delete_tree(
    connection: sqlite3.Connection,
    *,
    path: Path,
    created_at: str,
) -> None:
    """Record one best-effort directory deletion to run after DB commit."""

    resolved_path = path.expanduser().resolve()
    connection.execute(
        """
        INSERT INTO filesystem_operations(kind, source_path, target_path, created_at)
        VALUES (?, ?, ?, ?)
        """,
        ("delete_tree", str(resolved_path), None, created_at),
    )


def queue_move_tree(
    connection: sqlite3.Connection,
    *,
    source_path: Path,
    target_path: Path,
    created_at: str,
) -> None:
    """Record one directory move that must be replayed until it succeeds."""

    resolved_source_path = source_path.expanduser().resolve()
    resolved_target_path = target_path.expanduser().resolve()
    connection.execute(
        """
        INSERT INTO filesystem_operations(kind, source_path, target_path, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            "move_tree",
            str(resolved_source_path),
            str(resolved_target_path),
            created_at,
        ),
    )


def filesystem_operation_from_row(row: sqlite3.Row) -> FilesystemOperation:
    """Decode one persisted filesystem operation row."""

    return FilesystemOperation(
        id=int(row["id"]),
        kind=_filesystem_operation_kind(row["kind"]),
        source_path=Path(str(row["source_path"])).expanduser().resolve(),
        target_path=_optional_path(row["target_path"]),
        created_at=str(row["created_at"]),
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
