# src/rl_fzerox/core/training/session/model/algorithms.py
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from rl_fzerox.core.config.schema import TrainAppConfig, TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from sb3x import (
        HybridActionSAC,
        MaskableHybridActionPPO,
        MaskableHybridActionSAC,
        MaskableHybridRecurrentPPO,
        MaskableRecurrentPPO,
    )
    from stable_baselines3 import SAC

    PpoTrainingAlgorithmClass: TypeAlias = (
        type[MaskablePPO]
        | type[MaskableRecurrentPPO]
        | type[MaskableHybridActionPPO]
        | type[MaskableHybridRecurrentPPO]
    )
    SacTrainingAlgorithmClass: TypeAlias = (
        type[SAC] | type[HybridActionSAC] | type[MaskableHybridActionSAC]
    )
    TrainingAlgorithmClass: TypeAlias = PpoTrainingAlgorithmClass | SacTrainingAlgorithmClass
else:
    PpoTrainingAlgorithmClass: TypeAlias = type
    SacTrainingAlgorithmClass: TypeAlias = type
    TrainingAlgorithmClass: TypeAlias = type


def training_requires_action_masks(config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking."""

    return config.train.algorithm in TRAINING_ALGORITHMS.maskable


def resolve_effective_training_algorithm(
    *,
    train_config: TrainConfig,
) -> str:
    """Resolve the configured train.algorithm into the concrete algorithm used.

    `auto` is now a backwards-compatible alias for `maskable_ppo`. Recurrent
    training must be selected explicitly so the saved run config is unambiguous.
    """

    if train_config.algorithm == TRAINING_ALGORITHMS.auto:
        return TRAINING_ALGORITHMS.maskable_ppo
    return train_config.algorithm


def resolve_training_algorithm_class(algorithm: str) -> TrainingAlgorithmClass:
    try:
        if algorithm in TRAINING_ALGORITHMS.sac_family:
            return resolve_sac_training_algorithm_class(algorithm)
        return resolve_ppo_training_algorithm_class(algorithm)
    except ImportError as exc:
        if algorithm in TRAINING_ALGORITHMS.sb3x:
            raise RuntimeError(
                f"{algorithm} requires stable-baselines3, torch, and sb3x. "
                "Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc


def resolve_ppo_training_algorithm_class(algorithm: str) -> PpoTrainingAlgorithmClass:
    try:
        if algorithm == TRAINING_ALGORITHMS.maskable_recurrent_ppo:
            from sb3x import MaskableRecurrentPPO

            return MaskableRecurrentPPO
        if algorithm == TRAINING_ALGORITHMS.maskable_ppo:
            from sb3_contrib import MaskablePPO

            return MaskablePPO
        if algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
            from sb3x import MaskableHybridActionPPO

            return MaskableHybridActionPPO
        if algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
            from sb3x import MaskableHybridRecurrentPPO

            return MaskableHybridRecurrentPPO

        raise ValueError(f"Unsupported PPO-family training algorithm: {algorithm!r}")
    except ImportError as exc:
        if algorithm in TRAINING_ALGORITHMS.sb3x:
            raise RuntimeError(
                f"{algorithm} requires stable-baselines3, torch, and sb3x. "
                "Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc


def resolve_sac_training_algorithm_class(
    algorithm: str = TRAINING_ALGORITHMS.sac,
) -> SacTrainingAlgorithmClass:
    try:
        if algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_sac:
            from sb3x import MaskableHybridActionSAC

            return MaskableHybridActionSAC
        if algorithm == TRAINING_ALGORITHMS.hybrid_action_sac:
            from sb3x import HybridActionSAC

            return HybridActionSAC
        from stable_baselines3 import SAC

        return SAC
    except ImportError as exc:
        if algorithm in TRAINING_ALGORITHMS.sb3x:
            raise RuntimeError(
                f"{algorithm} requires stable-baselines3, torch, and sb3x. "
                "Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc
