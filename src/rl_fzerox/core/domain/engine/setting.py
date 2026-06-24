# src/rl_fzerox/core/domain/engine/setting.py
"""Canonical F-Zero X engine-slider values.

The in-game engine slider is not a decimal percent. Menu probing shows it has
129 representable positions, from ``0 / 128`` through ``128 / 128``. The game
HUD rounds that fraction to an ``ENG`` percent for display, so e.g. displayed
``90`` is actually slider step 115, or ``115 / 128 = 0.8984375``.

Project code should treat the integer slider step as the source of truth and
derive floats/percent labels only at display or native-RAM boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import floor


@dataclass(frozen=True, slots=True)
class EngineSliderSpec:
    """Canonical representable in-game engine slider range."""

    min_step: int = 0
    max_step: int = 128
    center_step: int = 64

    @property
    def step_count(self) -> int:
        return self.max_step - self.min_step + 1


ENGINE_SLIDER = EngineSliderSpec()


def clamp_engine_slider_step(value: int | float) -> int:
    """Clamp one value to the game-representable inclusive slider-step range."""

    return max(
        ENGINE_SLIDER.min_step,
        min(ENGINE_SLIDER.max_step, int(value)),
    )


def validate_engine_slider_step(
    value: int | float,
    *,
    label: str = "engine_setting_raw_value",
) -> int:
    """Return a valid slider step or raise a value error with a precise label."""

    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer slider step, got {value}")
    step = int(value)
    if not ENGINE_SLIDER.min_step <= step <= ENGINE_SLIDER.max_step:
        raise ValueError(
            f"{label} must be in [{ENGINE_SLIDER.min_step}, {ENGINE_SLIDER.max_step}], got {step}"
        )
    return step


def engine_slider_step_to_value(step: int | float) -> float:
    """Return the exact normalized game value for one slider step."""

    return validate_engine_slider_step(step) / float(ENGINE_SLIDER.max_step)


def engine_slider_step_to_percent(step: int | float) -> float:
    """Return the unrounded percent represented by one slider step."""

    return engine_slider_step_to_value(step) * 100.0


def engine_slider_step_to_display_percent(step: int | float) -> int:
    """Return the rounded ENG percent the game displays for one slider step."""

    return _round_half_up(engine_slider_step_to_percent(step))


def engine_percent_to_slider_step(percent: int | float) -> int:
    """Map a legacy/display percent to the nearest representable slider step."""

    normalized = max(0.0, min(100.0, float(percent))) / 100.0
    return clamp_engine_slider_step(_round_half_up(normalized * ENGINE_SLIDER.max_step))


def engine_value_to_slider_step(value: int | float) -> int:
    """Map a normalized engine float to the nearest representable slider step."""

    normalized = max(0.0, min(1.0, float(value)))
    return clamp_engine_slider_step(_round_half_up(normalized * ENGINE_SLIDER.max_step))


def engine_slider_steps(*, minimum: int, maximum: int) -> tuple[int, ...]:
    """Return inclusive canonical slider steps clamped to the game range."""

    lower = clamp_engine_slider_step(minimum)
    upper = clamp_engine_slider_step(maximum)
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    return tuple(range(lower, upper + 1))


def centered_engine_slider_buckets(
    *,
    minimum: int,
    maximum: int,
    side_count: int,
) -> tuple[int, ...]:
    """Return a center-anchored, evenly spaced engine bucket list.

    ``side_count`` is the number of buckets on each side of the neutral
    midpoint. A value of 5 therefore yields 11 total buckets: five slower
    buckets, neutral 50%, and five faster buckets.
    """

    lower = clamp_engine_slider_step(minimum)
    upper = clamp_engine_slider_step(maximum)
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    count = int(side_count)
    if count < 0:
        raise ValueError(f"engine bucket side_count must be >= 0, got {count}")
    if not lower <= ENGINE_SLIDER.center_step <= upper:
        raise ValueError("centered engine buckets require the engine range to include 50%")
    if count == 0:
        return (ENGINE_SLIDER.center_step,)
    values = _rounded_step_span(lower, ENGINE_SLIDER.center_step, count)[:-1] + _rounded_step_span(
        ENGINE_SLIDER.center_step, upper, count
    )
    expected_count = count * 2 + 1
    bucket_values = validate_engine_slider_bucket_values(values)
    if len(bucket_values) != expected_count:
        raise ValueError(
            "engine bucket side_count is too high for the selected range: "
            f"expected {expected_count} unique buckets, got {len(bucket_values)}"
        )
    return bucket_values


def validate_engine_slider_bucket_values(values: Iterable[int | float]) -> tuple[int, ...]:
    """Return sorted unique engine buckets or raise a precise value error."""

    buckets = tuple(
        validate_engine_slider_step(value, label="engine bucket raw value") for value in values
    )
    if not buckets:
        raise ValueError("engine bucket raw values must not be empty")
    if len(set(buckets)) != len(buckets):
        raise ValueError("engine bucket raw values must not contain duplicates")
    return tuple(sorted(buckets))


def _round_half_up(value: float) -> int:
    return int(floor(float(value) + 0.5))


def _rounded_step_span(start: int, end: int, interval_count: int) -> tuple[int, ...]:
    return tuple(
        _round_half_up(start + ((end - start) * index / float(interval_count)))
        for index in range(interval_count + 1)
    )
