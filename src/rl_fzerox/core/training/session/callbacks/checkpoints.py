# src/rl_fzerox/core/training/session/callbacks/checkpoints.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.config.schema import TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Resolved checkpoint cadence and retention policy for one training run."""

    step_interval: int | None
    rollout_interval: int | None
    save_latest: bool
    save_best: bool
    save_recent: bool
    recent_limit: int | None


def resolve_checkpoint_policy(train_config: TrainConfig) -> CheckpointPolicy:
    """Convert train config knobs into one runtime checkpoint policy."""

    rollout_interval = _rollout_checkpoint_interval(train_config)
    step_interval = None
    if rollout_interval is None:
        step_interval = max(1, train_config.save_freq // train_config.num_envs)
    return CheckpointPolicy(
        step_interval=step_interval,
        rollout_interval=rollout_interval,
        save_latest=train_config.save_latest_checkpoint,
        save_best=train_config.save_best_checkpoint,
        save_recent=train_config.save_recent_checkpoints,
        recent_limit=train_config.recent_checkpoint_limit,
    )


def _rollout_checkpoint_interval(train_config: TrainConfig) -> int | None:
    if train_config.algorithm in TRAINING_ALGORITHMS.sac_family:
        return None
    return train_config.checkpoint_every_rollouts
