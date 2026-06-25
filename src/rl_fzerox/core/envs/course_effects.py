# src/rl_fzerox/core/envs/course_effects.py
"""Helpers for interpreting course-effect bits from telemetry.

Reward, observation, and watch code use these predicates as the shared place for
raw surface-flag checks for refill, dirt, ice, and dash surfaces.
"""
from __future__ import annotations

from enum import IntEnum

from fzerox_emulator import FZeroXTelemetry


class CourseEffect(IntEnum):
    """Course-effect values stored in the low bits of the racer state flags."""

    NONE = 0
    PIT = 1
    DIRT = 2
    DASH = 3
    ICE = 4


GROUND_EFFECT_FEATURES: tuple[str, ...] = (
    "ground_pit",
    "ground_dash",
    "ground_dirt",
    "ground_ice",
)


def course_effect_raw(telemetry: FZeroXTelemetry | None) -> int:
    if telemetry is None:
        return int(CourseEffect.NONE)
    return int(telemetry.player.course_effect_raw)


def on_refill_surface(telemetry: FZeroXTelemetry | None) -> bool:
    return course_effect_raw(telemetry) == CourseEffect.PIT


def ground_effect_flags(telemetry: FZeroXTelemetry | None) -> tuple[float, float, float, float]:
    raw_effect = course_effect_raw(telemetry)
    return (
        1.0 if raw_effect == CourseEffect.PIT else 0.0,
        1.0 if raw_effect == CourseEffect.DASH else 0.0,
        1.0 if raw_effect == CourseEffect.DIRT else 0.0,
        1.0 if raw_effect == CourseEffect.ICE else 0.0,
    )
