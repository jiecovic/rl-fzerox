# src/rl_fzerox/core/domain/training_algorithms.py
"""Training algorithm names and capability groups shared across the repo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TrainAlgorithmName: TypeAlias = Literal[
    "maskable_ppo",
    "maskable_recurrent_ppo",
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
    "sac",
    "hybrid_action_sac",
    "maskable_hybrid_action_sac",
]


@dataclass(frozen=True, slots=True)
class TrainingAlgorithmRegistry:
    """Stable YAML names plus capability groups for supported trainers."""

    maskable_ppo: TrainAlgorithmName = "maskable_ppo"
    maskable_recurrent_ppo: TrainAlgorithmName = "maskable_recurrent_ppo"
    maskable_hybrid_action_ppo: TrainAlgorithmName = "maskable_hybrid_action_ppo"
    maskable_hybrid_recurrent_ppo: TrainAlgorithmName = "maskable_hybrid_recurrent_ppo"
    sac: TrainAlgorithmName = "sac"
    hybrid_action_sac: TrainAlgorithmName = "hybrid_action_sac"
    maskable_hybrid_action_sac: TrainAlgorithmName = "maskable_hybrid_action_sac"

    @property
    def default(self) -> TrainAlgorithmName:
        return self.maskable_ppo

    @property
    def maskable(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_ppo,
                self.maskable_recurrent_ppo,
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
                self.maskable_hybrid_action_sac,
            )
        )

    @property
    def recurrent(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_recurrent_ppo,
                self.maskable_hybrid_recurrent_ppo,
            )
        )

    @property
    def sb3x(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_recurrent_ppo,
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
                self.hybrid_action_sac,
                self.maskable_hybrid_action_sac,
            )
        )

    @property
    def sac_family(self) -> frozenset[TrainAlgorithmName]:
        return frozenset((self.sac, self.hybrid_action_sac, self.maskable_hybrid_action_sac))

    @property
    def full_model_policy(self) -> frozenset[TrainAlgorithmName]:
        return frozenset(
            (
                self.maskable_recurrent_ppo,
                self.maskable_hybrid_action_ppo,
                self.maskable_hybrid_recurrent_ppo,
                self.hybrid_action_sac,
                self.maskable_hybrid_action_sac,
            )
        )

TRAINING_ALGORITHMS = TrainingAlgorithmRegistry()
