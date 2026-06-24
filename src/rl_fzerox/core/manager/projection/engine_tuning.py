# src/rl_fzerox/core/manager/projection/engine_tuning.py
"""Manager-owned projection of adaptive engine-tuning runtime config."""

from __future__ import annotations

import json
from hashlib import blake2b

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
                "greedy_plateau_tolerance_seconds": (
                    vehicle.adaptive_engine_greedy_plateau_seconds
                ),
            }
        )
    if vehicle.adaptive_engine_tuner_backend == "bandit":
        payload.update(
            {
                "objective": vehicle.adaptive_engine_tuner_objective,
                "reward_fingerprint": _reward_fingerprint(config),
                "bucket_raw_values": vehicle.adaptive_engine_bandit_bucket_raw_values,
                "safe_finish_rate_threshold": (vehicle.adaptive_engine_safe_finish_rate_threshold),
                "min_finish_rate_observations": (
                    vehicle.adaptive_engine_min_finish_rate_observations
                ),
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
                "greedy_plateau_tolerance_seconds": (
                    vehicle.adaptive_engine_greedy_plateau_seconds
                ),
            }
        )
    return AdaptiveEngineTuningConfig.model_validate(payload)


def _reward_fingerprint(config: ManagedRunConfig) -> str:
    """Return a stable key for reward-dependent engine-tuner observations."""

    payload = {"reward": config.reward.model_dump(mode="json")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return blake2b(encoded, digest_size=12).hexdigest()
