# src/rl_fzerox/core/envs/actions/hybrid/layouts.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.envs.actions.base import DiscreteActionDimension


@dataclass(frozen=True, slots=True)
class HybridActionLayout:
    """Shape metadata for one hybrid continuous/discrete action adapter."""

    continuous_size: int
    dimensions: tuple[DiscreteActionDimension, ...]

    @property
    def discrete_size(self) -> int:
        return len(self.dimensions)


@dataclass(frozen=True, slots=True)
class LeanPrimitiveValues:
    """Reserved lean primitive branch values and the currently enabled subset."""

    count: int = 7
    enabled_default: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class PitchBucketValues:
    """Discrete left-stick Y buckets for airborne pitch control."""

    count: int = 5
    neutral_index: int = 2


LEAN_PRIMITIVES = LeanPrimitiveValues()
PITCH_BUCKETS = PitchBucketValues()
STEER_DRIVE_LEAN_LAYOUT = HybridActionLayout(
    continuous_size=2,
    dimensions=(DiscreteActionDimension("lean", 3),),
)
STEER_DRIVE_BOOST_LEAN_LAYOUT = HybridActionLayout(
    continuous_size=2,
    dimensions=(
        DiscreteActionDimension("lean", 3),
        DiscreteActionDimension("boost", 2),
    ),
)
STEER_GAS_BOOST_LEAN_LAYOUT = HybridActionLayout(
    continuous_size=1,
    dimensions=(
        DiscreteActionDimension("gas", 2),
        DiscreteActionDimension("boost", 2),
        DiscreteActionDimension("lean", 3),
    ),
)
STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT = HybridActionLayout(
    continuous_size=1,
    dimensions=(
        DiscreteActionDimension("gas", 2),
        DiscreteActionDimension("air_brake", 2),
        DiscreteActionDimension("boost", 2),
        DiscreteActionDimension("lean", 3),
    ),
)
STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT = HybridActionLayout(
    continuous_size=1,
    dimensions=(
        DiscreteActionDimension("gas", 2),
        DiscreteActionDimension("air_brake", 2),
        DiscreteActionDimension("boost", 2),
        DiscreteActionDimension("lean", 3),
        DiscreteActionDimension("pitch", PITCH_BUCKETS.count),
    ),
)
STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT = HybridActionLayout(
    continuous_size=2,
    dimensions=(
        DiscreteActionDimension("air_brake", 2),
        DiscreteActionDimension("boost", 2),
        DiscreteActionDimension("lean", 3),
        DiscreteActionDimension("pitch", PITCH_BUCKETS.count),
    ),
)
STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT = HybridActionLayout(
    continuous_size=3,
    dimensions=(
        DiscreteActionDimension("lean", LEAN_PRIMITIVES.count),
        DiscreteActionDimension("boost", 2),
    ),
)
