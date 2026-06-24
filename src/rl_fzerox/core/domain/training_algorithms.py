# src/rl_fzerox/core/domain/training_algorithms.py
"""Training algorithm names and capability groups shared across the repo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type TrainAlgorithmName = Literal[
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
]


@dataclass(frozen=True, slots=True)
class TrainingAlgorithmRegistry:
    """Stable runtime-spec names plus capability groups for supported trainers."""

    maskable_hybrid_action_ppo: TrainAlgorithmName = "maskable_hybrid_action_ppo"
    maskable_hybrid_recurrent_ppo: TrainAlgorithmName = "maskable_hybrid_recurrent_ppo"

    @property
    def default(self) -> TrainAlgorithmName:
        return self.maskable_hybrid_action_ppo

    @property
    def maskable(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
            )
        )

    @property
    def recurrent(self) -> frozenset[TrainAlgorithmName]:
        return frozenset((self.maskable_hybrid_recurrent_ppo,))

    @property
    def hybrid(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
            )
        )

    @property
    def sb3x(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
            )
        )

    @property
    def full_model_policy(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
            )
        )


TRAINING_ALGORITHMS = TrainingAlgorithmRegistry()
