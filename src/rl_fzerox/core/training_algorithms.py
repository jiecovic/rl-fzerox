# src/rl_fzerox/core/training_algorithms.py
"""String-backed training algorithm names shared by config, training, and watch."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

TrainAlgorithmName: TypeAlias = Literal[
    "auto",
    "ppo",
    "maskable_ppo",
    "maskable_recurrent_ppo",
    "hybrid_action_ppo",
    "hybrid_recurrent_ppo",
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
    "sac",
]

TRAIN_ALGORITHM_AUTO: Final[TrainAlgorithmName] = "auto"
TRAIN_ALGORITHM_PPO: Final[TrainAlgorithmName] = "ppo"
TRAIN_ALGORITHM_MASKABLE_PPO: Final[TrainAlgorithmName] = "maskable_ppo"
TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO: Final[TrainAlgorithmName] = "maskable_recurrent_ppo"
TRAIN_ALGORITHM_HYBRID_ACTION_PPO: Final[TrainAlgorithmName] = "hybrid_action_ppo"
TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO: Final[TrainAlgorithmName] = "hybrid_recurrent_ppo"
TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO: Final[TrainAlgorithmName] = (
    "maskable_hybrid_action_ppo"
)
TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO: Final[TrainAlgorithmName] = (
    "maskable_hybrid_recurrent_ppo"
)
TRAIN_ALGORITHM_SAC: Final[TrainAlgorithmName] = "sac"
DEFAULT_TRAIN_ALGORITHM: Final[TrainAlgorithmName] = TRAIN_ALGORITHM_MASKABLE_PPO


def _algorithm_set(*names: TrainAlgorithmName) -> frozenset[TrainAlgorithmName]:
    """Build typed immutable groups while keeping YAML/SB3-facing values as strings."""

    return frozenset(names)


LEGACY_PPO_ALGORITHMS: Final = _algorithm_set(
    TRAIN_ALGORITHM_AUTO,
    TRAIN_ALGORITHM_PPO,
)
MASKABLE_TRAINING_ALGORITHMS: Final = _algorithm_set(
    TRAIN_ALGORITHM_AUTO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
)
RECURRENT_TRAINING_ALGORITHMS: Final = _algorithm_set(
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
)
SB3X_TRAINING_ALGORITHMS: Final = _algorithm_set(
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
)
FULL_MODEL_POLICY_ALGORITHMS: Final = _algorithm_set(
    TRAIN_ALGORITHM_MASKABLE_RECURRENT_PPO,
    TRAIN_ALGORITHM_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_HYBRID_RECURRENT_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_ACTION_PPO,
    TRAIN_ALGORITHM_MASKABLE_HYBRID_RECURRENT_PPO,
)
SAVED_POLICY_ALGORITHMS: Final = _algorithm_set(
    *FULL_MODEL_POLICY_ALGORITHMS,
    TRAIN_ALGORITHM_MASKABLE_PPO,
    TRAIN_ALGORITHM_SAC,
)
