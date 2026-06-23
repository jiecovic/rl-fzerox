# src/rl_fzerox/core/manager/db/repositories/runs/drafts.py
"""Repository operations for run drafts and built-in templates."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.runs import RunDraftModel, RunTemplateModel
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import ManagedRunDraft


def assert_draft_name_available(
    session: Session,
    name: str,
    *,
    exclude_draft_id: str | None = None,
) -> None:
    """Reject draft names that would collide case-insensitively."""

    statement = select(RunDraftModel.id).where(func.lower(RunDraftModel.name) == name.lower())
    if exclude_draft_id is not None:
        statement = statement.where(RunDraftModel.id != exclude_draft_id)
    if session.scalar(statement.limit(1)) is not None:
        raise ManagerNameConflictError(kind="draft", name=name)


def insert_draft(
    session: Session,
    *,
    draft: ManagedRunDraft,
    config_snapshot_id: str,
) -> None:
    """Insert one editable draft row."""

    session.add(
        RunDraftModel(
            id=draft.id,
            name=draft.name,
            config_snapshot_id=config_snapshot_id,
            source_run_id=draft.source_run_id,
            source_artifact=draft.source_artifact,
            source_snapshot_dir=(
                None if draft.source_snapshot_dir is None else str(draft.source_snapshot_dir)
            ),
            source_num_timesteps=draft.source_num_timesteps,
            created_at=draft.created_at,
            updated_at=draft.updated_at,
        )
    )


def update_draft(
    session: Session,
    *,
    draft_id: str,
    name: str,
    config_snapshot_id: str,
    source_run_id: str | None,
    source_artifact: str | None,
    source_snapshot_dir: str | None,
    source_num_timesteps: int | None,
    updated_at: str,
) -> bool:
    """Update a draft row after a new config snapshot has been created."""

    draft = session.get(RunDraftModel, draft_id)
    if draft is None:
        return False
    draft.name = name
    draft.config_snapshot_id = config_snapshot_id
    draft.source_run_id = source_run_id
    draft.source_artifact = source_artifact
    draft.source_snapshot_dir = source_snapshot_dir
    draft.source_num_timesteps = source_num_timesteps
    draft.updated_at = updated_at
    return True


def upsert_template(
    session: Session,
    *,
    template_id: str,
    name: str,
    config_snapshot_id: str,
    created_at: str,
    updated_at: str,
) -> None:
    """Insert or refresh one built-in template row."""

    template = session.get(RunTemplateModel, template_id)
    if template is None:
        session.add(
            RunTemplateModel(
                id=template_id,
                name=name,
                config_snapshot_id=config_snapshot_id,
                created_at=created_at,
                updated_at=updated_at,
            )
        )
        return
    template.name = name
    template.config_snapshot_id = config_snapshot_id
    template.updated_at = updated_at
