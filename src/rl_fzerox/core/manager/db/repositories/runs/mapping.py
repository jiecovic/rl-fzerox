# src/rl_fzerox/core/manager/db/repositories/runs/mapping.py
"""ORM-to-domain mapping for managed run rows."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.domain.engine import ENGINE_SLIDER
from rl_fzerox.core.manager.db.models.metadata import LineageGroupModel
from rl_fzerox.core.manager.db.models.runs import RunModel
from rl_fzerox.core.manager.db.models.runtime import (
    RunCommandModel,
    RunRuntimeModel,
    RunWorkerModel,
)
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunRuntime,
    ManagedRunSummary,
    ManagedRunVehicleSummary,
)
from rl_fzerox.core.manager.registry.common import (
    optional_source_artifact,
    run_command,
    run_status,
)
from rl_fzerox.core.manager.storage.serialization import load_config_json


def managed_run_from_model(session: Session, run: RunModel) -> ManagedRun:
    """Build the public run dataclass from ORM rows in the current transaction."""

    lineage_id = _required_lineage_id(run)
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

    lineage_id = _required_lineage_id(run)
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


def _required_lineage_id(run: RunModel) -> str:
    if run.lineage_id:
        return run.lineage_id
    raise RuntimeError(f"manager DB is not current: run {run.id} is missing lineage_id")


def _string_value(value: object, *, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _bounded_int(value: object, *, fallback: int) -> int:
    if not isinstance(value, int | float):
        return fallback
    return max(ENGINE_SLIDER.min_step, min(ENGINE_SLIDER.max_step, int(value)))
