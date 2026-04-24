# src/rl_fzerox/core/training/session/model/validation.py
from __future__ import annotations

from rl_fzerox.core.config.schema import PolicyConfig, TrainAppConfig
from rl_fzerox.core.domain.action_adapters import HYBRID_ACTION_ADAPTERS, SAC_ACTION_ADAPTERS
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.training.session.model.replay import SUPPORTED_LAZY_REPLAY_STACK_MODES


def validate_training_algorithm_config(config: TrainAppConfig) -> None:
    """Reject incompatible algorithm/config combinations before training starts."""

    if config.train.algorithm == TRAINING_ALGORITHMS.sac:
        _validate_sac_training_config(config)
        return
    if config.train.algorithm == TRAINING_ALGORITHMS.hybrid_action_sac:
        _validate_hybrid_action_sac_training_config(config)
        return
    if config.train.algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_sac:
        _validate_maskable_hybrid_sac_config(config)
        return
    if config.train.algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
        _validate_maskable_hybrid_ppo_config(config)
        return
    if config.train.algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
        _validate_maskable_recurrent_hybrid_ppo_config(config)


def validate_masking_configuration(*, train_env, effective_algorithm: str) -> None:
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


def _validate_sac_training_config(config: TrainAppConfig) -> None:
    action_config = config.env.action.runtime()
    if action_config.name not in SAC_ACTION_ADAPTERS:
        raise RuntimeError(
            "SAC requires a continuous steer-drive action adapter so the action space is Box"
        )
    if action_config.mask_overrides is not None:
        raise RuntimeError("SAC does not support env.action.mask; use the continuous adapter")
    if config.curriculum.enabled:
        raise RuntimeError("SAC does not support curriculum stages")
    _validate_sac_lazy_replay_support(config)


def _validate_hybrid_action_sac_training_config(config: TrainAppConfig) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Hybrid action SAC",
    )
    action_config = config.env.action.runtime()
    if action_config.mask_overrides is not None:
        raise RuntimeError(
            "Hybrid action SAC is not maskable yet; do not configure env.action.mask"
        )
    if config.curriculum.enabled:
        raise RuntimeError("Hybrid action SAC does not support curriculum stages yet")
    _validate_sac_lazy_replay_support(config)


def _validate_maskable_hybrid_ppo_config(config: TrainAppConfig) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Maskable hybrid action PPO",
    )


def _validate_maskable_hybrid_sac_config(config: TrainAppConfig) -> None:
    _validate_hybrid_action_adapter(
        config,
        algorithm_label="Maskable hybrid action SAC",
    )
    _validate_sac_lazy_replay_support(config)


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
    if action_config.name not in HYBRID_ACTION_ADAPTERS:
        raise RuntimeError(
            f"{algorithm_label} requires a hybrid steer-drive action adapter "
            "so the action space is Dict(continuous=Box, discrete=MultiDiscrete)"
        )


def _validate_sac_lazy_replay_support(config: TrainAppConfig) -> None:
    if config.env.observation.mode != "image_state" or not config.train.optimize_memory_usage:
        return

    stack_mode = config.env.observation.stack_mode
    if stack_mode not in SUPPORTED_LAZY_REPLAY_STACK_MODES:
        supported = ", ".join(sorted(SUPPORTED_LAZY_REPLAY_STACK_MODES))
        raise RuntimeError(
            "SAC lazy replay does not support "
            f"observation.stack_mode={stack_mode!r}; use one of: {supported}"
        )
