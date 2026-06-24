# src/rl_fzerox/core/envs/observations/state/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fzerox_emulator.arrays import Float32Array
from rl_fzerox.core.domain.observations import (
    ActionHistoryControlName,
)

type ObservationMode = Literal["image", "image_state"]
type ActionHistoryControl = ActionHistoryControlName


@dataclass(frozen=True, slots=True)
class ObservationStateDefaults:
    """Default scalar-observation settings shared by env, watch, and tests."""

    builtin_course_count: int = 24
    speed_normalizer_kph: float = 1_500.0
    lean_tap_guard_frames: int = 15
    lateral_velocity_normalizer: float = 32.0
    sliding_lateral_velocity_threshold: float = 8.0


OBSERVATION_STATE_DEFAULTS = ObservationStateDefaults()


@dataclass(frozen=True, slots=True)
class StateFeature:
    """One scalar policy-state feature and its observation-space upper bound."""

    name: str
    high: float
    low: float = 0.0


@dataclass(frozen=True, slots=True)
class StateVectorSpec:
    """Ordered scalar state schema appended to image observations."""

    features: tuple[StateFeature, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def count(self) -> int:
        return len(self.features)

    def high_array(self) -> Float32Array:
        return np.array([feature.high for feature in self.features], dtype=np.float32)

    def low_array(self) -> Float32Array:
        return np.array([feature.low for feature in self.features], dtype=np.float32)
