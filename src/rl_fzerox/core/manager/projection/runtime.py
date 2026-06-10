# src/rl_fzerox/core/manager/projection/runtime.py
from __future__ import annotations

from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_artifacts_from_state,
    restore_generated_x_cup_entries_from_state,
)
from rl_fzerox.core.manager.store import ManagerStore
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


def restore_managed_runtime_track_sampling(
    config: TrainAppConfig,
    *,
    store: ManagerStore,
    run_id: str,
    include_artifacts: bool,
) -> TrainAppConfig:
    """Apply manager-owned mutable track-sampling runtime state to a train config."""

    restored = restore_generated_x_cup_entries_from_state(
        config,
        state=store.get_run_track_sampling_state(run_id),
    )
    if not include_artifacts:
        return restored
    return restore_generated_x_cup_artifacts_from_state(
        restored,
        artifacts=store.get_run_track_sampling_artifacts(run_id),
    )
