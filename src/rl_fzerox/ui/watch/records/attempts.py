# src/rl_fzerox/ui/watch/records/attempts.py
"""Attempt counters and completion aggregation for Watch track records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry


@dataclass(frozen=True, slots=True)
class TrackAttemptStats:
    attempts: int = 0
    finishes: int = 0
    completion_samples: int = 0
    completion_sum: float = 0.0
    best_completion: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, int | float] | None) -> TrackAttemptStats:
        return cls(
            attempts=_stats_int(values, "attempts"),
            finishes=_stats_int(values, "finishes"),
            completion_samples=_stats_int(values, "completion_samples"),
            completion_sum=_stats_float(values, "completion_sum"),
            best_completion=_stats_float(values, "best_completion"),
        )

    def as_mapping(self) -> dict[str, int | float]:
        return {
            "attempts": self.attempts,
            "finishes": self.finishes,
            "completion_samples": self.completion_samples,
            "completion_sum": self.completion_sum,
            "best_completion": self.best_completion,
        }

    def update(
        self,
        *,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        episode_done: bool,
    ) -> TrackAttemptStats:
        if not episode_done:
            return self
        completion = _episode_completion_fraction(info, telemetry)
        completion_samples = self.completion_samples
        completion_sum = self.completion_sum
        best_completion = self.best_completion
        if completion is not None:
            completion_samples += 1
            completion_sum += completion
            best_completion = max(best_completion, completion)
        return TrackAttemptStats(
            attempts=self.attempts + 1,
            finishes=self.finishes + (1 if info.get("termination_reason") == "finished" else 0),
            completion_samples=completion_samples,
            completion_sum=completion_sum,
            best_completion=best_completion,
        )


def _episode_completion_fraction(
    info: Mapping[str, object],
    telemetry: FZeroXTelemetry | None,
) -> float | None:
    value = info.get("episode_completion_fraction")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _clamped_fraction(float(value))
    if info.get("termination_reason") == "finished":
        return 1.0
    if telemetry is None:
        return None

    course_length = float(telemetry.course_length)
    total_lap_count = int(telemetry.total_lap_count)
    total_race_distance = course_length * total_lap_count
    if total_race_distance <= 0.0:
        return None
    return _clamped_fraction(float(telemetry.player.race_distance) / total_race_distance)


def _stats_int(values: Mapping[str, int | float] | None, key: str) -> int:
    if values is None:
        return 0
    value = values.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return max(0, int(value))
    return 0


def _stats_float(values: Mapping[str, int | float] | None, key: str) -> float:
    if values is None:
        return 0.0
    value = values.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return max(0.0, float(value))
    return 0.0


def _clamped_fraction(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
