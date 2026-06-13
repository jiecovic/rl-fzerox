# src/rl_fzerox/core/manager/projection/engine_tuning.py
"""Manager-owned projection of adaptive engine-tuning runtime config."""

from __future__ import annotations

from rl_fzerox.core.engine_tuning.config import (
    engine_tuning_episode_horizon_prior_seconds,
    engine_tuning_uncertainty_scale_seconds,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


def adaptive_engine_tuning_config(config: ManagedRunConfig) -> AdaptiveEngineTuningConfig:
    """Return runtime engine-tuning config derived from the manager run spec."""

    vehicle = config.vehicle
    prior_finish_time_seconds = engine_tuning_episode_horizon_prior_seconds(
        max_episode_steps=config.environment.max_episode_steps,
        action_repeat=config.action.action_repeat,
    )
    uncertainty_scale_seconds = engine_tuning_uncertainty_scale_seconds(
        prior_finish_time_seconds=prior_finish_time_seconds
    )
    payload: dict[str, object] = {
        "enabled": vehicle.engine_mode == "adaptive_tuner",
        "min_raw_value": vehicle.engine_setting_min_raw_value,
        "max_raw_value": vehicle.engine_setting_max_raw_value,
        "backend": vehicle.adaptive_engine_tuner_backend,
        "prior_finish_time_seconds": prior_finish_time_seconds,
        "uniform_exploration": vehicle.adaptive_engine_uniform_exploration,
    }
    if vehicle.adaptive_engine_tuner_backend == "gaussian_process":
        payload.update(
            {
                "stat_decay": vehicle.adaptive_engine_stat_decay,
                "exploration_scale": uncertainty_scale_seconds,
            }
        )
    if vehicle.adaptive_engine_tuner_backend == "mlp_ensemble":
        payload.update(
            {
                "ensemble_members": vehicle.adaptive_engine_ensemble_members,
                "randomized_prior_seconds": uncertainty_scale_seconds,
                "hidden_dim": vehicle.adaptive_engine_mlp_hidden_dim,
                "training_steps": vehicle.adaptive_engine_mlp_training_steps,
                "learning_rate": vehicle.adaptive_engine_mlp_learning_rate,
                "bootstrap_keep_probability": (
                    vehicle.adaptive_engine_mlp_bootstrap_keep_probability
                ),
                "warmup_successes": vehicle.adaptive_engine_mlp_warmup_successes,
            }
        )
    return AdaptiveEngineTuningConfig.model_validate(payload)
