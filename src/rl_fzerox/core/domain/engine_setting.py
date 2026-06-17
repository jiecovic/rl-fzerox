# src/rl_fzerox/core/domain/engine_setting.py
"""Canonical F-Zero X engine-slider values.

The in-game engine slider is not a decimal percent. Menu probing shows it has
129 representable positions, from ``0 / 128`` through ``128 / 128``. The game
HUD rounds that fraction to an ``ENG`` percent for display, so e.g. displayed
``90`` is actually slider step 115, or ``115 / 128 = 0.8984375``.

Project code should treat the integer slider step as the source of truth and
derive floats/percent labels only at display or native-RAM boundaries.
"""

from __future__ import annotations

from math import floor

ENGINE_SLIDER_STEP_MIN = 0
ENGINE_SLIDER_STEP_MAX = 128
ENGINE_SLIDER_STEP_CENTER = 64
ENGINE_SLIDER_STEP_COUNT = ENGINE_SLIDER_STEP_MAX - ENGINE_SLIDER_STEP_MIN + 1


def clamp_engine_slider_step(value: int | float) -> int:
    """Clamp one value to the game-representable inclusive slider-step range."""

    return max(
        ENGINE_SLIDER_STEP_MIN,
        min(ENGINE_SLIDER_STEP_MAX, int(value)),
    )


def validate_engine_slider_step(
    value: int | float,
    *,
    label: str = "engine_setting_raw_value",
) -> int:
    """Return a valid slider step or raise a value error with a precise label."""

    step = int(value)
    if not ENGINE_SLIDER_STEP_MIN <= step <= ENGINE_SLIDER_STEP_MAX:
        raise ValueError(
            f"{label} must be in [{ENGINE_SLIDER_STEP_MIN}, {ENGINE_SLIDER_STEP_MAX}], got {step}"
        )
    return step


def engine_slider_step_to_value(step: int | float) -> float:
    """Return the exact normalized game value for one slider step."""

    return validate_engine_slider_step(step) / float(ENGINE_SLIDER_STEP_MAX)


def engine_slider_step_to_percent(step: int | float) -> float:
    """Return the unrounded percent represented by one slider step."""

    return engine_slider_step_to_value(step) * 100.0


def engine_slider_step_to_display_percent(step: int | float) -> int:
    """Return the rounded ENG percent the game displays for one slider step."""

    return _round_half_up(engine_slider_step_to_percent(step))


def engine_percent_to_slider_step(percent: int | float) -> int:
    """Map a legacy/display percent to the nearest representable slider step."""

    normalized = max(0.0, min(100.0, float(percent))) / 100.0
    return clamp_engine_slider_step(_round_half_up(normalized * ENGINE_SLIDER_STEP_MAX))


def engine_value_to_slider_step(value: int | float) -> int:
    """Map a normalized engine float to the nearest representable slider step."""

    normalized = max(0.0, min(1.0, float(value)))
    return clamp_engine_slider_step(_round_half_up(normalized * ENGINE_SLIDER_STEP_MAX))


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
    slider_spacing: int,
) -> tuple[int, ...]:
    """Return centered bucket steps, mirrored around the neutral slider midpoint."""

    lower = clamp_engine_slider_step(minimum)
    upper = clamp_engine_slider_step(maximum)
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    step = max(1, int(slider_spacing))
    values: set[int] = {lower, upper}
    offset = 0
    while True:
        low = ENGINE_SLIDER_STEP_CENTER - offset
        high = ENGINE_SLIDER_STEP_CENTER + offset
        if lower <= low <= upper:
            values.add(low)
        if offset != 0 and lower <= high <= upper:
            values.add(high)
        if low < lower and high > upper:
            break
        offset += step
    return tuple(sorted(values))


def _round_half_up(value: float) -> int:
    return int(floor(float(value) + 0.5))
