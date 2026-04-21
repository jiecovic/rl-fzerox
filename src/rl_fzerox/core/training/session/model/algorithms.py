# src/rl_fzerox/core/training/session/model/algorithms.py
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from rl_fzerox.core.config.schema import TrainAppConfig, TrainConfig
from rl_fzerox.core.domain.training_algorithms import (
    MASKABLE_TRAINING_ALGORITHMS,
    SB3X_TRAINING_ALGORITHMS,
    TRAIN_ALGORITHM_AUTO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_SAC,
)

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from sb3x import (
        MaskableHybridActionPPO,
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
    SacTrainingAlgorithmClass: TypeAlias = type[SAC]
    TrainingAlgorithmClass: TypeAlias = PpoTrainingAlgorithmClass | SacTrainingAlgorithmClass
else:
    PpoTrainingAlgorithmClass: TypeAlias = type
    SacTrainingAlgorithmClass: TypeAlias = type
    TrainingAlgorithmClass: TypeAlias = type


def training_requires_action_masks(config: TrainAppConfig) -> bool:
    """Return whether the current env stack depends on action masking."""

    return config.train.algorithm in MASKABLE_TRAINING_ALGORITHMS


def resolve_effective_training_algorithm(
    *,
    train_config: TrainConfig,
) -> str:
    """Resolve the configured train.algorithm into the concrete algorithm used.

    `auto` is now a backwards-compatible alias for `maskable_ppo`. Recurrent
    training must be selected explicitly so the saved run config is unambiguous.
    """

    if train_config.algorithm == TRAIN_ALGORITHM_AUTO:
        return TRAIN_ALGORITHM_MASKABLE_PPO
    return train_config.algorithm


def resolve_training_algorithm_class(algorithm: str) -> TrainingAlgorithmClass:
    try:
        if algorithm == TRAIN_ALGORITHM_SAC:
            from stable_baselines3 import SAC

            return SAC
        return resolve_ppo_training_algorithm_class(algorithm)
    except ImportError as exc:
        if algorithm in SB3X_TRAINING_ALGORITHMS:
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
        if algorithm == TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO:
            from sb3x import MaskableRecurrentPPO

            return MaskableRecurrentPPO
        if algorithm == TRAIN_ALGORITHM_MASKABLE_PPO:
            from sb3_contrib import MaskablePPO

            return MaskablePPO
        if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO:
            from sb3x import MaskableHybridActionPPO

            return MaskableHybridActionPPO
        if algorithm == TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO:
            from sb3x import MaskableHybridRecurrentPPO

            return MaskableHybridRecurrentPPO

        raise ValueError(f"Unsupported PPO-family training algorithm: {algorithm!r}")
    except ImportError as exc:
        if algorithm in SB3X_TRAINING_ALGORITHMS:
            raise RuntimeError(
                f"{algorithm} requires stable-baselines3, torch, and sb3x. "
                "Install train deps and then install sb3x in "
                "the active environment."
            ) from exc
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc


def resolve_sac_training_algorithm_class() -> SacTrainingAlgorithmClass:
    try:
        from stable_baselines3 import SAC

        return SAC
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc
