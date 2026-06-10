# src/rl_fzerox/core/manager/projection/watch.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_artifacts_from_state,
    restore_generated_x_cup_entries_from_state,
)
from rl_fzerox.core.manager.store import ManagerStore
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.runs import (
    continue_run_paths,
    materialize_train_run_config,
)


def managed_watch_train_config(
    *,
    db_path: Path,
    run_id: str,
) -> tuple[Path, TrainAppConfig, int | None]:
    """Resolve one manager-owned run into a materialized runtime training config."""

    store = ManagerStore(db_path)
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"managed run not found: {run_id}")
    lineage_frame_offset = lineage_frame_offset_for_run(store, run)
    train_config = build_managed_train_app_config(
        run.config,
        run_id=run.id,
        run_dir=run.run_dir,
    )
    train_config = restore_generated_x_cup_entries_from_state(
        train_config,
        state=store.get_run_track_sampling_state(run.id),
    )
    train_config = restore_generated_x_cup_artifacts_from_state(
        train_config,
        artifacts=store.get_run_track_sampling_artifacts(run.id),
    )
    return (
        run.run_dir,
        materialize_train_run_config(
            train_config,
            run_paths=continue_run_paths(run.run_dir),
        ),
        lineage_frame_offset,
    )


def lineage_frame_offset_for_run(store: ManagerStore, run: ManagedRun) -> int | None:
    """Return emulator frames completed before this run's local checkpoint timeline."""

    total_frames = 0
    current_run = run
    while current_run.parent_run_id is not None:
        parent_run = store.get_run(current_run.parent_run_id)
        if parent_run is None:
            return None
        source_steps = current_run.source_num_timesteps
        if source_steps is None:
            source_steps = current_run.lineage_step_offset - parent_run.lineage_step_offset
        if source_steps < 0:
            return None
        total_frames += source_steps * max(1, int(parent_run.config.action.action_repeat))
        current_run = parent_run
    return total_frames
