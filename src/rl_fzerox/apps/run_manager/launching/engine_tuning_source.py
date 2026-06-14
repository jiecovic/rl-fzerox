# src/rl_fzerox/apps/run_manager/launching/engine_tuning_source.py
"""Prepare adaptive engine-tuning sidecars for managed fork sources."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from rl_fzerox.core.engine_tuning import (
    BanditEngineTunerSettings,
    OrderedEngineTuner,
    save_engine_tuning_runtime_state,
)
from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
from rl_fzerox.core.manager.projection.engine_tuning import adaptive_engine_tuning_config
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import (
    engine_tuning_checkpoint_path,
    engine_tuning_model_path,
    load_engine_tuning_checkpoint_state,
)

EngineTuningSourceAction: TypeAlias = Literal["convert", "discard"]


def prepare_engine_tuning_fork_source(
    *,
    config: ManagedRunConfig,
    source_dir: Path,
    artifact: Literal["latest", "best"],
    action: EngineTuningSourceAction,
) -> None:
    """Rewrite or remove tuner history in one copied fork-source checkpoint."""

    adaptive_config = adaptive_engine_tuning_config(config)
    if not adaptive_config.enabled:
        return
    settings = engine_tuner_settings(adaptive_config)
    if not isinstance(settings, BanditEngineTunerSettings):
        return

    policy_path = resolve_policy_artifact_path(source_dir, artifact=artifact)
    state_path = engine_tuning_checkpoint_path(policy_path)
    model_path = engine_tuning_model_path(policy_path)
    if action == "discard":
        _unlink_if_exists(state_path)
        _unlink_if_exists(model_path)
        return

    state = load_engine_tuning_checkpoint_state(policy_path)
    if state is None:
        _unlink_if_exists(model_path)
        return

    canonical_state = OrderedEngineTuner(settings=settings, state=state).state
    save_engine_tuning_runtime_state(
        state_path,
        canonical_state,
        model_path=model_path,
    )


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
