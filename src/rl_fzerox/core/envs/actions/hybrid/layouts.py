# src/rl_fzerox/core/envs/actions/hybrid/layouts.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.envs.actions.base import DiscreteActionDimension


@dataclass(frozen=True, slots=True)
class HybridActionLayout:
    """Shape metadata for one configured hybrid action layout."""

    continuous_size: int
    dimensions: tuple[DiscreteActionDimension, ...]

    @property
    def discrete_size(self) -> int:
        return len(self.dimensions)
@dataclass(frozen=True, slots=True)
class PitchBucketValues:
    """Discrete left-stick Y buckets for airborne pitch control."""

    count: int = 5
    neutral_index: int = 2


PITCH_BUCKETS = PitchBucketValues()
