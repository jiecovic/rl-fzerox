# src/rl_fzerox/core/manager/registry/runs/runtime.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.manager.registry.rows import run_from_row, run_select_sql
from rl_fzerox.core.training.session.callbacks.track_sampling.persistence import (
    load_track_sampling_runtime_state_json,
    track_sampling_runtime_state_json,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeState,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def clear_run_runtime(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (run_id,))


def clear_run_track_sampling_state(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute("DELETE FROM run_track_sampling_state WHERE run_id = ?", (run_id,))


def get_run_track_sampling_state(
    store: ManagerStore,
    run_id: str,
) -> TrackSamplingRuntimeState | None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        row = connection.execute(
            """
            SELECT state_json
            FROM run_track_sampling_state
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return load_track_sampling_runtime_state_json(str(row["state_json"]))


def upsert_run_track_sampling_state(
    store: ManagerStore,
    *,
    run_id: str,
    state: TrackSamplingRuntimeState,
    updated_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    state_json = track_sampling_runtime_state_json(state)
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO run_track_sampling_state(run_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (run_id, state_json, updated_at or utc_now()),
        )


def upsert_run_runtime(
    store: ManagerStore,
    *,
    run_id: str,
    total_timesteps: int,
    num_timesteps: int,
    progress_fraction: float,
    updated_at: str,
    fps: float | None = None,
    episode_reward_mean: float | None = None,
    episode_length_mean: float | None = None,
    approx_kl: float | None = None,
    entropy_loss: float | None = None,
    value_loss: float | None = None,
    policy_gradient_loss: float | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO run_runtime(
                run_id,
                total_timesteps,
                num_timesteps,
                progress_fraction,
                updated_at,
                fps,
                episode_reward_mean,
                episode_length_mean,
                approx_kl,
                entropy_loss,
                value_loss,
                policy_gradient_loss
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                total_timesteps = excluded.total_timesteps,
                num_timesteps = excluded.num_timesteps,
                progress_fraction = excluded.progress_fraction,
                updated_at = excluded.updated_at,
                fps = excluded.fps,
                episode_reward_mean = excluded.episode_reward_mean,
                episode_length_mean = excluded.episode_length_mean,
                approx_kl = excluded.approx_kl,
                entropy_loss = excluded.entropy_loss,
                value_loss = excluded.value_loss,
                policy_gradient_loss = excluded.policy_gradient_loss
            """,
            (
                run_id,
                total_timesteps,
                num_timesteps,
                progress_fraction,
                updated_at,
                fps,
                episode_reward_mean,
                episode_length_mean,
                approx_kl,
                entropy_loss,
                value_loss,
                policy_gradient_loss,
            ),
        )


def append_run_event(
    store: ManagerStore,
    *,
    run_id: str,
    kind: str,
    message: str,
    created_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    event_at = created_at or utc_now()
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, event_at, kind, message),
        )


def update_run_fork_source(
    store: ManagerStore,
    *,
    run_id: str,
    source_snapshot_dir: Path,
    source_num_timesteps: int,
    lineage_step_offset: int | None = None,
) -> ManagedRun | None:
    store.initialize()
    rebuilt_at = utc_now()
    with store._connect() as connection:
        row = connection.execute(
            """
            UPDATE runs
            SET
                source_snapshot_dir = ?,
                source_num_timesteps = ?,
                lineage_step_offset = COALESCE(?, lineage_step_offset)
            WHERE id = ?
            RETURNING id
            """,
            (
                str(source_snapshot_dir.expanduser().resolve()),
                source_num_timesteps,
                lineage_step_offset,
                run_id,
            ),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                rebuilt_at,
                "fork_source_rebuilt",
                "recreated pinned fork source snapshot",
            ),
        )
        selected_row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if selected_row is None else run_from_row(selected_row)
