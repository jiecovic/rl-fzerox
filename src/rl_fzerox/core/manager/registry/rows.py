# src/rl_fzerox/core/manager/registry/rows.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunRuntime,
    ManagedRunSummary,
    ManagedRunTemplate,
)
from rl_fzerox.core.manager.registry.common import (
    optional_float,
    optional_int,
    optional_path,
    optional_source_artifact,
    optional_str,
    run_command,
    run_status,
)
from rl_fzerox.core.manager.storage.serialization import load_config_json


@dataclass(frozen=True, slots=True)
class RunWorkerLease:
    run_id: str
    launch_token: str
    pid: int
    launched_at: str
    heartbeat_at: str


def run_from_row(row: sqlite3.Row) -> ManagedRun:
    return ManagedRun(
        id=str(row["id"]),
        name=str(row["name"]),
        status=run_status(row["status"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        run_dir=Path(str(row["run_dir"])),
        lineage_id=str(row["lineage_id"] or row["id"]),
        parent_run_id=optional_str(row["parent_run_id"]),
        source_run_id=optional_str(row["source_run_id"]),
        source_artifact=optional_source_artifact(row["source_artifact"]),
        source_snapshot_dir=optional_path(row["source_snapshot_dir"]),
        source_num_timesteps=optional_int(row["source_num_timesteps"]),
        created_at=str(row["created_at"]),
        lineage_step_offset=int(row["lineage_step_offset"]),
        lineage_groups=lineage_groups_from_row(row),
        started_at=optional_str(row["started_at"]),
        stopped_at=optional_str(row["stopped_at"]),
        worker_heartbeat_at=optional_str(row["worker_heartbeat_at"]),
        runtime=runtime_from_row(row),
        pending_command=run_command(row["pending_command"]),
    )


def run_summary_from_row(row: sqlite3.Row) -> ManagedRunSummary:
    action_repeat = row["config_action_repeat"]
    return ManagedRunSummary(
        id=str(row["id"]),
        name=str(row["name"]),
        status=run_status(row["status"]),
        config_hash=str(row["config_hash"]),
        action_repeat=max(1, int(action_repeat)) if isinstance(action_repeat, int | float) else 1,
        lineage_id=str(row["lineage_id"] or row["id"]),
        parent_run_id=optional_str(row["parent_run_id"]),
        source_run_id=optional_str(row["source_run_id"]),
        source_artifact=optional_source_artifact(row["source_artifact"]),
        source_num_timesteps=optional_int(row["source_num_timesteps"]),
        created_at=str(row["created_at"]),
        lineage_step_offset=int(row["lineage_step_offset"]),
        lineage_groups=lineage_groups_from_row(row),
        started_at=optional_str(row["started_at"]),
        stopped_at=optional_str(row["stopped_at"]),
        worker_heartbeat_at=optional_str(row["worker_heartbeat_at"]),
        runtime=runtime_from_row(row),
        pending_command=run_command(row["pending_command"]),
    )


def runtime_from_row(row: sqlite3.Row) -> ManagedRunRuntime | None:
    if not isinstance(row["runtime_updated_at"], str):
        return None
    return ManagedRunRuntime(
        total_timesteps=int(row["runtime_total_timesteps"]),
        num_timesteps=int(row["runtime_num_timesteps"]),
        progress_fraction=float(row["runtime_progress_fraction"]),
        updated_at=row["runtime_updated_at"],
        fps=optional_float(row["runtime_fps"]),
        episode_reward_mean=optional_float(row["runtime_episode_reward_mean"]),
        episode_length_mean=optional_float(row["runtime_episode_length_mean"]),
        approx_kl=optional_float(row["runtime_approx_kl"]),
        entropy_loss=optional_float(row["runtime_entropy_loss"]),
        value_loss=optional_float(row["runtime_value_loss"]),
        policy_gradient_loss=optional_float(row["runtime_policy_gradient_loss"]),
    )


def lineage_groups_from_row(row: sqlite3.Row) -> tuple[str, ...]:
    value = row["lineage_groups"]
    if not isinstance(value, str) or value == "":
        return ()
    return tuple(part for part in value.split("\n") if part)


def template_from_row(row: sqlite3.Row) -> ManagedRunTemplate:
    return ManagedRunTemplate(
        id=str(row["id"]),
        name=str(row["name"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def draft_from_row(row: sqlite3.Row) -> ManagedRunDraft:
    return ManagedRunDraft(
        id=str(row["id"]),
        name=str(row["name"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        source_run_id=optional_str(row["source_run_id"]),
        source_artifact=optional_source_artifact(row["source_artifact"]),
        source_snapshot_dir=optional_path(row["source_snapshot_dir"]),
        source_num_timesteps=optional_int(row["source_num_timesteps"]),
    )


def run_worker_lease_from_row(row: sqlite3.Row) -> RunWorkerLease:
    return RunWorkerLease(
        run_id=str(row["run_id"]),
        launch_token=str(row["launch_token"]),
        pid=int(row["pid"]),
        launched_at=str(row["launched_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
    )


def run_select_sql(
    *,
    where_clause: str = "",
    order_clause: str = "",
) -> str:
    return f"""
        SELECT
            runs.id,
            runs.name,
            runs.status,
            config_snapshots.config_json,
            config_snapshots.config_hash,
            runs.run_dir,
            runs.lineage_id,
            (
                SELECT group_concat(group_name, char(10))
                FROM (
                    SELECT group_name
                    FROM lineage_groups
                    WHERE lineage_groups.lineage_id = runs.lineage_id
                    ORDER BY group_name COLLATE NOCASE
                )
            ) AS lineage_groups,
            runs.lineage_step_offset,
            runs.parent_run_id,
            runs.source_run_id,
            runs.source_artifact,
            runs.source_snapshot_dir,
            runs.source_num_timesteps,
            runs.created_at,
            runs.started_at,
            runs.stopped_at,
            run_workers.heartbeat_at AS worker_heartbeat_at,
            run_runtime.total_timesteps AS runtime_total_timesteps,
            run_runtime.num_timesteps AS runtime_num_timesteps,
            run_runtime.progress_fraction AS runtime_progress_fraction,
            run_runtime.updated_at AS runtime_updated_at,
            run_runtime.fps AS runtime_fps,
            run_runtime.episode_reward_mean AS runtime_episode_reward_mean,
            run_runtime.episode_length_mean AS runtime_episode_length_mean,
            run_runtime.approx_kl AS runtime_approx_kl,
            run_runtime.entropy_loss AS runtime_entropy_loss,
            run_runtime.value_loss AS runtime_value_loss,
            run_runtime.policy_gradient_loss AS runtime_policy_gradient_loss,
            run_commands.command AS pending_command
        FROM runs
        JOIN config_snapshots
            ON config_snapshots.id = runs.config_snapshot_id
        LEFT JOIN run_runtime
            ON run_runtime.run_id = runs.id
        LEFT JOIN run_workers
            ON run_workers.run_id = runs.id
        LEFT JOIN run_commands
            ON run_commands.run_id = runs.id
        {where_clause}
        {order_clause}
    """


def run_summary_select_sql(
    *,
    where_clause: str = "",
    order_clause: str = "",
) -> str:
    return f"""
        SELECT
            runs.id,
            runs.name,
            runs.status,
            config_snapshots.config_hash,
            CAST(json_extract(config_snapshots.config_json, '$.action.action_repeat') AS INTEGER)
                AS config_action_repeat,
            runs.lineage_id,
            (
                SELECT group_concat(group_name, char(10))
                FROM (
                    SELECT group_name
                    FROM lineage_groups
                    WHERE lineage_groups.lineage_id = runs.lineage_id
                    ORDER BY group_name COLLATE NOCASE
                )
            ) AS lineage_groups,
            runs.lineage_step_offset,
            runs.parent_run_id,
            runs.source_run_id,
            runs.source_artifact,
            runs.source_num_timesteps,
            runs.created_at,
            runs.started_at,
            runs.stopped_at,
            run_workers.heartbeat_at AS worker_heartbeat_at,
            run_runtime.total_timesteps AS runtime_total_timesteps,
            run_runtime.num_timesteps AS runtime_num_timesteps,
            run_runtime.progress_fraction AS runtime_progress_fraction,
            run_runtime.updated_at AS runtime_updated_at,
            run_runtime.fps AS runtime_fps,
            run_runtime.episode_reward_mean AS runtime_episode_reward_mean,
            run_runtime.episode_length_mean AS runtime_episode_length_mean,
            run_runtime.approx_kl AS runtime_approx_kl,
            run_runtime.entropy_loss AS runtime_entropy_loss,
            run_runtime.value_loss AS runtime_value_loss,
            run_runtime.policy_gradient_loss AS runtime_policy_gradient_loss,
            run_commands.command AS pending_command
        FROM runs
        JOIN config_snapshots
            ON config_snapshots.id = runs.config_snapshot_id
        LEFT JOIN run_runtime
            ON run_runtime.run_id = runs.id
        LEFT JOIN run_workers
            ON run_workers.run_id = runs.id
        LEFT JOIN run_commands
            ON run_commands.run_id = runs.id
        {where_clause}
        {order_clause}
    """
