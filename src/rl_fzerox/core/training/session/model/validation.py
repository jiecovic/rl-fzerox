# src/rl_fzerox/core/training/session/model/validation.py
from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainAppConfig


def validate_training_algorithm_config(config: TrainAppConfig) -> None:
    """Reject incompatible algorithm/config combinations before training starts."""

    if config.train.algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
        _validate_maskable_hybrid_ppo_config(config)
        return
    if config.train.algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
        _validate_maskable_recurrent_hybrid_ppo_config(config)


def validate_masking_configuration(*, train_env: VecEnv, effective_algorithm: str) -> None:
    if effective_algorithm not in TRAINING_ALGORITHMS.maskable:
        return

    if not hasattr(train_env, "env_method"):
        raise RuntimeError("Mask-aware algorithms require a vector env exposing env_method()")
    if not train_env.has_attr("action_masks"):
        raise RuntimeError("Mask-aware algorithms require env.action_masks() support")


def validate_recurrent_configuration_alignment(
    *,
    effective_algorithm: str,
    policy_config: PolicyConfig,
) -> None:
    recurrent_enabled = policy_config.recurrent.enabled
    if recurrent_enabled and effective_algorithm not in TRAINING_ALGORITHMS.recurrent:
        raise RuntimeError("Recurrent policy config requires a recurrent train.algorithm")
    if not recurrent_enabled and effective_algorithm in TRAINING_ALGORITHMS.recurrent:
        raise RuntimeError(f"{effective_algorithm} requires policy.recurrent.enabled=true")


def validate_auxiliary_state_configuration(
    *,
    train_env: VecEnv,
    policy_config: PolicyConfig,
    effective_algorithm: str,
) -> None:
    auxiliary_state = policy_config.auxiliary_state
    if not auxiliary_state.enabled:
        return
    if not isinstance(train_env.observation_space, spaces.Dict):
        raise RuntimeError(
            "policy.auxiliary_state requires observation.mode=image_state "
            "so hidden aux targets can ride alongside dict observations"
        )
    if effective_algorithm == TRAINING_ALGORITHMS.maskable_ppo:
        raise RuntimeError(
            "policy.auxiliary_state is not wired for maskable_ppo yet; "
            "use a recurrent or hybrid sb3x PPO variant"
        )


def _validate_maskable_hybrid_ppo_config(config: TrainAppConfig) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Maskable hybrid action PPO",
    )


def _validate_maskable_recurrent_hybrid_ppo_config(
    config: TrainAppConfig,
) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Maskable hybrid recurrent PPO",
    )


def _validate_hybrid_action_adapter(
    config: TrainAppConfig,
    *,
    algorithm_label: str,
) -> None:
    action_config = config.env.action.runtime()
    if action_config.name != "configured_hybrid":
        raise RuntimeError(
            f"{algorithm_label} requires the configured_hybrid action layout "
            "so the action space is Dict(continuous=Box, discrete=MultiDiscrete)"
        )
