# src/rl_fzerox/core/manager/db/repositories/runs.py
"""Repository operations for run, draft, template, and event rows."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.metadata import LineageGroupModel
from rl_fzerox.core.manager.db.models.runs import (
    RunDraftModel,
    RunEventModel,
    RunModel,
    RunTemplateModel,
)
from rl_fzerox.core.manager.db.models.runtime import (
    RunCommandModel,
    RunRuntimeModel,
    RunWorkerModel,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import ManagedRun, ManagedRunDraft, ManagedRunRuntime
from rl_fzerox.core.manager.registry.common import (
    optional_source_artifact,
    run_command,
    run_status,
)
from rl_fzerox.core.manager.storage.serialization import load_config_json


def resolve_lineage_id(
    session: Session,
    *,
    explicit_lineage_id: str | None,
    parent_run_id: str | None,
    source_run_id: str | None,
    fallback_run_id: str,
) -> str:
    """Resolve the lineage id for a new run from its parent/source run."""

    if explicit_lineage_id is not None:
        return explicit_lineage_id
    parent_id = parent_run_id or source_run_id
    if parent_id is None:
        return fallback_run_id
    parent_run = session.get(RunModel, parent_id)
    if parent_run is None:
        return fallback_run_id
    return parent_run.lineage_id or parent_id


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


def insert_run(
    session: Session,
    *,
    run: ManagedRun,
    config_snapshot_id: str,
) -> None:
    """Insert one launched run identity row."""

    session.add(
        RunModel(
            id=run.id,
            name=run.name,
            status=run.status,
            config_snapshot_id=config_snapshot_id,
            run_dir=str(run.run_dir),
            lineage_id=run.lineage_id,
            lineage_step_offset=run.lineage_step_offset,
            parent_run_id=run.parent_run_id,
            source_run_id=run.source_run_id,
            source_artifact=run.source_artifact,
            source_snapshot_dir=(
                None if run.source_snapshot_dir is None else str(run.source_snapshot_dir)
            ),
            source_num_timesteps=run.source_num_timesteps,
            created_at=run.created_at,
            started_at=run.started_at,
            stopped_at=run.stopped_at,
        )
    )


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


def append_run_event(
    session: Session,
    *,
    run_id: str,
    created_at: str,
    kind: str,
    message: str,
) -> None:
    """Append one event row for a run."""

    session.add(
        RunEventModel(
            run_id=run_id,
            created_at=created_at,
            kind=kind,
            message=message,
        )
    )


def managed_run_from_model(session: Session, run: RunModel) -> ManagedRun:
    """Build the public run dataclass from ORM rows in the current transaction."""

    lineage_id = run.lineage_id or run.id
    lineage_groups = tuple(
        session.scalars(
            select(LineageGroupModel.group_name)
            .where(LineageGroupModel.lineage_id == lineage_id)
            .order_by(LineageGroupModel.group_name)
        )
    )
    runtime = session.get(RunRuntimeModel, run.id)
    worker = session.get(RunWorkerModel, run.id)
    pending_command = session.get(RunCommandModel, run.id)
    return ManagedRun(
        id=run.id,
        name=run.name,
        status=run_status(run.status),
        config=load_config_json(run.config_snapshot.config_json),
        config_hash=run.config_snapshot.config_hash,
        run_dir=Path(run.run_dir),
        lineage_id=lineage_id,
        lineage_groups=lineage_groups,
        lineage_step_offset=run.lineage_step_offset,
        parent_run_id=run.parent_run_id,
        source_run_id=run.source_run_id,
        source_artifact=optional_source_artifact(run.source_artifact),
        source_snapshot_dir=(
            None if run.source_snapshot_dir is None else Path(run.source_snapshot_dir)
        ),
        source_num_timesteps=run.source_num_timesteps,
        created_at=run.created_at,
        started_at=run.started_at,
        stopped_at=run.stopped_at,
        worker_heartbeat_at=None if worker is None else worker.heartbeat_at,
        runtime=None if runtime is None else _managed_runtime_from_model(runtime),
        pending_command=run_command(None if pending_command is None else pending_command.command),
    )


def _managed_runtime_from_model(runtime: RunRuntimeModel) -> ManagedRunRuntime:
    return ManagedRunRuntime(
        total_timesteps=runtime.total_timesteps,
        num_timesteps=runtime.num_timesteps,
        progress_fraction=runtime.progress_fraction,
        updated_at=runtime.updated_at,
        fps=runtime.fps,
        episode_reward_mean=runtime.episode_reward_mean,
        episode_length_mean=runtime.episode_length_mean,
        approx_kl=runtime.approx_kl,
        entropy_loss=runtime.entropy_loss,
        value_loss=runtime.value_loss,
        policy_gradient_loss=runtime.policy_gradient_loss,
    )
