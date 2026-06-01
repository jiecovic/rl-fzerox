# src/rl_fzerox/core/manager/registry/runs/runtime.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.models import RunEventModel, RunModel, RunRuntimeModel
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.registry.common import utc_now

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def clear_run_runtime(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        runtime = session.get(RunRuntimeModel, run_id)
        if runtime is not None:
            session.delete(runtime)


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
    with store._orm_session() as session:
        runtime = session.get(RunRuntimeModel, run_id)
        if runtime is None:
            session.add(
                RunRuntimeModel(
                    run_id=run_id,
                    total_timesteps=total_timesteps,
                    num_timesteps=num_timesteps,
                    progress_fraction=progress_fraction,
                    updated_at=updated_at,
                    fps=fps,
                    episode_reward_mean=episode_reward_mean,
                    episode_length_mean=episode_length_mean,
                    approx_kl=approx_kl,
                    entropy_loss=entropy_loss,
                    value_loss=value_loss,
                    policy_gradient_loss=policy_gradient_loss,
                )
            )
            return
        runtime.total_timesteps = total_timesteps
        runtime.num_timesteps = num_timesteps
        runtime.progress_fraction = progress_fraction
        runtime.updated_at = updated_at
        runtime.fps = fps
        runtime.episode_reward_mean = episode_reward_mean
        runtime.episode_length_mean = episode_length_mean
        runtime.approx_kl = approx_kl
        runtime.entropy_loss = entropy_loss
        runtime.value_loss = value_loss
        runtime.policy_gradient_loss = policy_gradient_loss


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
    with store._orm_session() as session:
        session.add(
            RunEventModel(
                run_id=run_id,
                created_at=event_at,
                kind=kind,
                message=message,
            )
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
    with store._orm_session() as session:
        run = session.get(RunModel, run_id)
        if run is None:
            return None
        run.source_snapshot_dir = str(source_snapshot_dir.expanduser().resolve())
        run.source_num_timesteps = source_num_timesteps
        if lineage_step_offset is not None:
            run.lineage_step_offset = lineage_step_offset
        session.add(
            RunEventModel(
                run_id=run_id,
                created_at=rebuilt_at,
                kind="fork_source_rebuilt",
                message="recreated pinned fork source snapshot",
            )
        )
    return store.get_run(run_id)
