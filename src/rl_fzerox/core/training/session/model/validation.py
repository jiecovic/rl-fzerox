# src/rl_fzerox/core/training/session/model/validation.py
from __future__ import annotations

from rl_fzerox.core.action_adapters import HYBRID_ACTION_ADAPTERS, SAC_ACTION_ADAPTERS
from rl_fzerox.core.config.schema import PolicyConfig, TrainAppConfig
from rl_fzerox.core.training_algorithms import (
    MASKABLE_TRAINING_ALGORITHMS,
    RECURRENT_TRAINING_ALGORITHMS,
    TRAIN_ALGORITHM_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_PPO,
    TRAIN_ALGORITHM_SAC,
)


def validate_training_algorithm_config(config: TrainAppConfig) -> None:
    """Reject incompatible algorithm/config combinations before training starts."""

    if config.train.algorithm == TRAIN_ALGORITHM_PPO:
        raise RuntimeError(
            "Plain PPO training is no longer supported. "
            f"Use `train.algorithm={TRAIN_ALGORITHM_MASKABLE_PPO}` or "
            f"`{TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO}`."
        )
    if config.train.algorithm == TRAIN_ALGORITHM_SAC:
        _validate_sac_training_config(config)
        return
    if config.train.algorithm == TRAIN_ALGORITHM_HYBRID_ACTION_PPO:
        _validate_hybrid_action_ppo_training_config(config)
        return
    if config.train.algorithm == TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO:
        _validate_hybrid_recurrent_ppo_training_config(config)
        return
    if config.train.algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO:
        _validate_maskable_hybrid_action_ppo_training_config(config)
        return
    if config.train.algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO:
        _validate_maskable_hybrid_recurrent_ppo_training_config(config)


def validate_masking_configuration(*, train_env, effective_algorithm: str) -> None:
    if effective_algorithm not in MASKABLE_TRAINING_ALGORITHMS:
        return

    if not hasattr(train_env, "env_method"):
        raise RuntimeError("Maskable PPO requires a vector env exposing env_method()")
    if not train_env.has_attr("action_masks"):
        raise RuntimeError("Maskable PPO requires env.action_masks() support")


def validate_recurrent_configuration_alignment(
    *,
    effective_algorithm: str,
    policy_config: PolicyConfig,
) -> None:
    recurrent_enabled = policy_config.recurrent.enabled
    if recurrent_enabled and effective_algorithm not in RECURRENT_TRAINING_ALGORITHMS:
        raise RuntimeError(
            "Recurrent policy config requires a recurrent train.algorithm"
        )
    if not recurrent_enabled and effective_algorithm in RECURRENT_TRAINING_ALGORITHMS:
        raise RuntimeError(
            f"{effective_algorithm} requires policy.recurrent.enabled=true"
        )


def _validate_sac_training_config(config: TrainAppConfig) -> None:
    if config.env.action.name not in SAC_ACTION_ADAPTERS:
        raise RuntimeError(
            "SAC requires a continuous steer-drive action adapter so the action space is Box"
        )
    if config.env.action.mask is not None:
        raise RuntimeError("SAC does not support env.action.mask; use the continuous adapter")
    if config.curriculum.enabled:
        raise RuntimeError("SAC does not support action-mask curriculum stages")
    if config.env.observation.mode == "image_state" and config.train.optimize_memory_usage:
        raise RuntimeError(
            "SAC optimize_memory_usage is not supported with Dict image_state observations"
        )


def _validate_hybrid_action_ppo_training_config(config: TrainAppConfig) -> None:
    _validate_unmasked_hybrid_action_ppo_training_config(
        config,
        algorithm_label="Hybrid action PPO",
    )


def _validate_hybrid_recurrent_ppo_training_config(config: TrainAppConfig) -> None:
    _validate_unmasked_hybrid_action_ppo_training_config(
        config,
        algorithm_label="Hybrid recurrent PPO",
    )


def _validate_unmasked_hybrid_action_ppo_training_config(
    config: TrainAppConfig,
    *,
    algorithm_label: str,
) -> None:
    _validate_hybrid_action_adapter(config, algorithm_label=algorithm_label)
    if config.env.action.mask is not None:
        raise RuntimeError(f"{algorithm_label} does not support env.action.mask yet")
    if config.curriculum.enabled:
        raise RuntimeError(f"{algorithm_label} does not support action-mask curriculum stages")


def _validate_maskable_hybrid_action_ppo_training_config(config: TrainAppConfig) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Maskable hybrid action PPO",
    )


def _validate_maskable_hybrid_recurrent_ppo_training_config(
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
    if config.env.action.name not in HYBRID_ACTION_ADAPTERS:
        raise RuntimeError(
            f"{algorithm_label} requires a hybrid steer-drive action adapter "
            "so the action space is Dict(continuous=Box, discrete=MultiDiscrete)"
        )
