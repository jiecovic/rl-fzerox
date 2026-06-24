# src/rl_fzerox/core/evaluation/engine_tuning.py
"""Evaluation-time engine-tuning reset configuration."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.engine_tuning.contexts import engine_tuning_contexts_for_track_sampling
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.evaluation.env_control import (
    set_engine_tuning_sampler,
    set_engine_tuning_selection,
)
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state


def configure_evaluation_engine_tuning(
    env: object,
    config: TrainAppConfig,
    *,
    policy_path: Path,
) -> None:
    """Install frozen greedy engine-tuning choices for checkpoint evaluation."""

    track_sampling = config.env.track_sampling
    if not track_sampling.enabled or not track_sampling.engine_tuning.enabled:
        return

    contexts = engine_tuning_contexts_for_track_sampling(track_sampling)
    if not contexts:
        return

    state = load_engine_tuning_checkpoint_state(policy_path)
    controller = EngineTuningTrainingController(track_sampling.engine_tuning, state=state)
    set_engine_tuning_sampler(env, controller.reset_sampler_snapshot(contexts))
    set_engine_tuning_selection(env, "greedy")
