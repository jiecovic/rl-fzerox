# src/rl_fzerox/core/manager/db/repositories/filesystem.py
"""Repository operations for deferred manager filesystem mutations."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.artifacts.filesystem import (
    FilesystemOperation,
    filesystem_operation_from_values,
)
from rl_fzerox.core.manager.db.models.metadata import FilesystemOperationModel


def queue_delete_tree(
    session: Session,
    *,
    path: Path,
    created_at: str,
) -> None:
    """Persist one best-effort directory deletion to drain after commit."""

    session.add(
        FilesystemOperationModel(
            kind="delete_tree",
            source_path=str(path.expanduser().resolve()),
            target_path=None,
            created_at=created_at,
        )
    )


def list_filesystem_operations(session: Session) -> tuple[FilesystemOperation, ...]:
    """Return queued filesystem operations in replay order."""

    rows = tuple(
        session.scalars(
            select(FilesystemOperationModel).order_by(FilesystemOperationModel.id.asc())
        )
    )
    return tuple(_filesystem_operation_from_model(row) for row in rows)


def delete_filesystem_operation(session: Session, operation_id: int) -> None:
    """Remove a completed filesystem operation from the queue."""

    session.execute(
        delete(FilesystemOperationModel).where(FilesystemOperationModel.id == operation_id)
    )


def _filesystem_operation_from_model(
    model: FilesystemOperationModel,
) -> FilesystemOperation:
    return filesystem_operation_from_values(
        operation_id=model.id,
        kind=model.kind,
        source_path=model.source_path,
        target_path=model.target_path,
        created_at=model.created_at,
    )
