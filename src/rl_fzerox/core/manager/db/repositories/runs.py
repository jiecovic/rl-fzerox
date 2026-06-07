# src/rl_fzerox/core/manager/db/repositories/runs.py
"""Repository operations for run, draft, template, and event rows."""

from __future__ import annotations

import json
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
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunRuntime,
    ManagedRunSummary,
    ManagedRunVehicleSummary,
    RunStatus,
)
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


def get_managed_run(session: Session, run_id: str) -> ManagedRun | None:
    """Return one managed run by id."""

    run = session.get(RunModel, run_id)
    return None if run is None else managed_run_from_model(session, run)


def list_managed_runs(session: Session) -> tuple[ManagedRun, ...]:
    """Return all managed runs in manager display order."""

    runs = tuple(
        session.scalars(select(RunModel).order_by(RunModel.created_at.desc(), RunModel.id.desc()))
    )
    return tuple(managed_run_from_model(session, run) for run in runs)


def list_visible_managed_run_summaries(session: Session) -> tuple[ManagedRunSummary, ...]:
    """Return run-list summaries for launched runs."""

    runs = tuple(
        session.scalars(
            select(RunModel)
            .where(RunModel.status != "created")
            .order_by(RunModel.created_at.desc(), RunModel.id.desc())
        )
    )
    return tuple(managed_run_summary_from_model(session, run) for run in runs)


def list_recent_managed_run_events(
    session: Session,
    run_ids: tuple[str, ...],
    *,
    limit_per_run: int,
) -> dict[str, tuple[ManagedRunEvent, ...]]:
    """Return the most recent events per requested run id."""

    if not run_ids:
        return {}
    events = tuple(
        session.scalars(
            select(RunEventModel)
            .where(RunEventModel.run_id.in_(run_ids))
            .order_by(RunEventModel.created_at.desc(), RunEventModel.id.desc())
        )
    )
    events_by_run_id: dict[str, list[ManagedRunEvent]] = {run_id: [] for run_id in run_ids}
    for event in events:
        run_events = events_by_run_id.setdefault(event.run_id, [])
        if len(run_events) >= limit_per_run:
            continue
        run_events.append(
            ManagedRunEvent(
                run_id=event.run_id,
                created_at=event.created_at,
                kind=event.kind,
                message=event.message,
            )
        )
    return {
        event_run_id: tuple(run_events)
        for event_run_id, run_events in events_by_run_id.items()
        if run_events
    }


def set_run_status(
    session: Session,
    *,
    run_id: str,
    status: RunStatus,
    message: str,
    started_at: str | None,
    stopped_at: str | None,
    event_at: str,
) -> ManagedRun | None:
    """Update one run status and append the corresponding event."""

    run = session.get(RunModel, run_id)
    if run is None:
        return None
    run.status = status
    if started_at is not None:
        run.started_at = started_at
    run.stopped_at = stopped_at
    if status != "running":
        worker = session.get(RunWorkerModel, run_id)
        if worker is not None:
            session.delete(worker)
    append_run_event(session, run_id=run_id, created_at=event_at, kind=status, message=message)
    session.flush()
    return managed_run_from_model(session, run)


def rename_run(
    session: Session,
    *,
    run_id: str,
    name: str,
    renamed_at: str,
) -> ManagedRun | None:
    """Rename one run and append a manager event."""

    run = session.get(RunModel, run_id)
    if run is None:
        return None
    run.name = name
    append_run_event(
        session,
        run_id=run_id,
        created_at=renamed_at,
        kind="renamed",
        message=f"run renamed to {name}",
    )
    session.flush()
    return managed_run_from_model(session, run)


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


def managed_run_summary_from_model(session: Session, run: RunModel) -> ManagedRunSummary:
    """Build the lightweight run-list dataclass from ORM rows."""

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
    return ManagedRunSummary(
        id=run.id,
        name=run.name,
        status=run_status(run.status),
        config_hash=run.config_snapshot.config_hash,
        action_repeat=_action_repeat_from_config_json(run.config_snapshot.config_json),
        vehicle_setup=_vehicle_summary_from_config_json(run.config_snapshot.config_json),
        lineage_id=lineage_id,
        lineage_groups=lineage_groups,
        lineage_step_offset=run.lineage_step_offset,
        parent_run_id=run.parent_run_id,
        source_run_id=run.source_run_id,
        source_artifact=optional_source_artifact(run.source_artifact),
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


def _action_repeat_from_config_json(config_json: str) -> int:
    try:
        loaded = json.loads(config_json)
    except json.JSONDecodeError:
        return 1
    if not isinstance(loaded, dict):
        return 1
    action = loaded.get("action")
    if not isinstance(action, dict):
        return 1
    action_repeat = action.get("action_repeat")
    if not isinstance(action_repeat, int | float):
        return 1
    return max(1, int(action_repeat))


def _vehicle_summary_from_config_json(config_json: str) -> ManagedRunVehicleSummary:
    try:
        loaded = json.loads(config_json)
    except json.JSONDecodeError:
        loaded = {}
    vehicle = loaded.get("vehicle") if isinstance(loaded, dict) else None
    if not isinstance(vehicle, dict):
        vehicle = {}
    selected_vehicle_ids = vehicle.get("selected_vehicle_ids")
    if isinstance(selected_vehicle_ids, list):
        vehicle_ids = tuple(value for value in selected_vehicle_ids if isinstance(value, str))
    else:
        vehicle_ids = ()
    return ManagedRunVehicleSummary(
        selection_mode=_string_value(vehicle.get("selection_mode"), fallback="fixed"),
        selected_vehicle_ids=vehicle_ids or ("blue_falcon",),
        engine_mode=_string_value(vehicle.get("engine_mode"), fallback="fixed"),
        engine_setting_raw_value=_bounded_int(
            vehicle.get("engine_setting_raw_value"),
            fallback=50,
        ),
        engine_setting_min_raw_value=_bounded_int(
            vehicle.get("engine_setting_min_raw_value"),
            fallback=20,
        ),
        engine_setting_max_raw_value=_bounded_int(
            vehicle.get("engine_setting_max_raw_value"),
            fallback=80,
        ),
    )


def _string_value(value: object, *, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _bounded_int(value: object, *, fallback: int) -> int:
    if not isinstance(value, int | float):
        return fallback
    return max(0, min(100, int(value)))
