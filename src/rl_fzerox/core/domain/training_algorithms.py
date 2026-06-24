# src/rl_fzerox/core/domain/training_algorithms.py
"""Training algorithm names and capability groups shared across the repo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

type TrainAlgorithmName = Literal[
    "maskable_hybrid_action_ppo",
    "maskable_hybrid_recurrent_ppo",
]


@dataclass(frozen=True, slots=True)
class TrainingAlgorithmSpec:
    """Capabilities for one stable train.algorithm value."""

    name: TrainAlgorithmName
    maskable: bool
    recurrent: bool
    hybrid: bool
    sb3x: bool
    full_model_policy: bool


_TRAINING_ALGORITHM_SPECS: tuple[TrainingAlgorithmSpec, ...] = (
    TrainingAlgorithmSpec(
        name="maskable_hybrid_action_ppo",
        maskable=True,
        recurrent=False,
        hybrid=True,
        sb3x=True,
        full_model_policy=True,
    ),
    TrainingAlgorithmSpec(
        name="maskable_hybrid_recurrent_ppo",
        maskable=True,
        recurrent=True,
        hybrid=True,
        sb3x=True,
        full_model_policy=True,
    ),
)


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
        return _capability_group(lambda spec: spec.maskable)

    @property
    def recurrent(self) -> frozenset[TrainAlgorithmName]:
        return _capability_group(lambda spec: spec.recurrent)

    @property
    def hybrid(self) -> frozenset[TrainAlgorithmName]:
        return _capability_group(lambda spec: spec.hybrid)

    @property
    def sb3x(self) -> frozenset[TrainAlgorithmName]:
        return _capability_group(lambda spec: spec.sb3x)

    @property
    def full_model_policy(self) -> frozenset[TrainAlgorithmName]:
        return _capability_group(lambda spec: spec.full_model_policy)


def _capability_group(
    predicate: Callable[[TrainingAlgorithmSpec], bool],
) -> frozenset[TrainAlgorithmName]:
    return frozenset(spec.name for spec in _TRAINING_ALGORITHM_SPECS if predicate(spec))


TRAINING_ALGORITHMS = TrainingAlgorithmRegistry()
