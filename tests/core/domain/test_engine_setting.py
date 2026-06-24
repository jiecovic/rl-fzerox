# tests/core/domain/test_engine_setting.py
from __future__ import annotations

import pytest

from rl_fzerox.core.domain.engine import (
    ENGINE_SLIDER,
    centered_engine_slider_buckets,
    engine_percent_to_slider_step,
    engine_slider_step_to_display_percent,
    engine_slider_step_to_value,
    engine_value_to_slider_step,
    validate_engine_slider_bucket_values,
    validate_engine_slider_step,
)


def test_engine_slider_spec_matches_game_slider_range() -> None:
    assert ENGINE_SLIDER.min_step == 0
    assert ENGINE_SLIDER.center_step == 64
    assert ENGINE_SLIDER.max_step == 128
    assert ENGINE_SLIDER.step_count == 129


def test_engine_slider_percent_and_value_conversions_use_raw_steps() -> None:
    assert engine_slider_step_to_value(64) == 0.5
    assert engine_slider_step_to_display_percent(115) == 90
    assert engine_percent_to_slider_step(90) == 115
    assert engine_value_to_slider_step(0.5) == 64


def test_validate_engine_slider_step_rejects_non_integral_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="must be an integer slider step"):
        validate_engine_slider_step(64.5)
    with pytest.raises(ValueError, match=r"must be in \[0, 128\]"):
        validate_engine_slider_step(129)


def test_centered_engine_slider_buckets_are_unique_centered_steps() -> None:
    assert centered_engine_slider_buckets(minimum=0, maximum=128, side_count=2) == (
        0,
        32,
        64,
        96,
        128,
    )
    with pytest.raises(ValueError, match="include 50%"):
        centered_engine_slider_buckets(minimum=65, maximum=128, side_count=2)


def test_validate_engine_slider_bucket_values_sorts_and_rejects_duplicates() -> None:
    assert validate_engine_slider_bucket_values((80, 40, 64)) == (40, 64, 80)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        validate_engine_slider_bucket_values((64, 64))
