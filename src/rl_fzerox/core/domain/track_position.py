# src/rl_fzerox/core/domain/track_position.py
"""Shared track-position feature metadata and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class TrackPositionFeatureSpec:
    """One track-position feature with its shared normalization behavior."""

    name: str
    normalizer: float

    def normalize(self, raw_value: float) -> float:
        clamped = max(float(raw_value), 0.0)
        return max(0.0, min(clamped / self.normalizer, 1.0))


@lru_cache(maxsize=1)
def height_above_ground_feature() -> TrackPositionFeatureSpec:
    return TrackPositionFeatureSpec(
        name="track_position.height_above_ground_norm",
        normalizer=1_000.0,
    )
