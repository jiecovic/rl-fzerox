# src/rl_fzerox/core/training/session/model/algorithms.py
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS, TrainAlgorithmName
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig, TrainConfig

if TYPE_CHECKING:
    from sb3x import (
        MaskableHybridActionPPO,
        MaskableHybridRecurrentPPO,
    )

    PpoTrainingAlgorithmClass: TypeAlias = (
        type[MaskableHybridActionPPO] | type[MaskableHybridRecurrentPPO]
    )
    TrainingAlgorithmClass: TypeAlias = PpoTrainingAlgorithmClass
else:
    PpoTrainingAlgorithmClass: TypeAlias = type
    TrainingAlgorithmClass: TypeAlias = type


def training_requires_action_masks(config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking."""

    return config.train.algorithm in TRAINING_ALGORITHMS.maskable


def resolve_effective_training_algorithm(
    *,
    train_config: TrainConfig,
) -> TrainAlgorithmName:
    """Resolve the configured train.algorithm into the concrete algorithm used."""

    return train_config.algorithm


def resolve_training_algorithm_class(algorithm: str) -> TrainingAlgorithmClass:
    try:
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
