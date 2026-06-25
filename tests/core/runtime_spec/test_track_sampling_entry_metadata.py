# tests/core/runtime_spec/test_track_sampling_entry_metadata.py
"""Tests for flat track-sampling entries and their typed metadata views."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig


def test_track_sampling_entry_metadata_views_keep_flat_serialized_shape() -> None:
    entry = TrackSamplingEntryConfig(
        id="x_cup_slot_1",
        course_id="x_cup_1234abcd",
        course_name="X Cup 1234abcd",
        course_index=X_CUP_COURSE.course_index,
        mode=X_CUP_COURSE.race_mode,
        source_vehicle="blue_falcon",
        source_course_index=X_CUP_COURSE.course_index,
        source_gp_difficulty="master",
        source_engine_setting_raw_value=60,
        baseline_variant_index=2,
        baseline_variant_count=4,
        baseline_variant_seed=123,
        alt_baseline_id="alt-a",
        alt_baseline_label="frame 100",
        alt_baseline_source_entry_id="x_cup_slot_1_base",
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=456,
        generated_course_hash="abcd1234",
        generated_course_slot=0,
        generated_course_generation=3,
        generated_course_segment_count=38,
        generated_course_length=61_743.98,
    )

    source_setup = entry.source_setup_metadata()
    baseline_variant = entry.baseline_variant_metadata()
    alt_baseline = entry.alt_baseline_metadata()
    generated_course = entry.generated_course_metadata()

    assert source_setup.course_index == X_CUP_COURSE.course_index
    assert source_setup.gp_difficulty == "master"
    assert baseline_variant is not None
    assert baseline_variant.seed == 123
    assert alt_baseline is not None
    assert alt_baseline.source_entry_id == "x_cup_slot_1_base"
    assert generated_course is not None
    assert generated_course.course_hash == "abcd1234"

    data = entry.model_dump(mode="json", exclude_none=True)

    assert data["generated_course_hash"] == "abcd1234"
    assert data["alt_baseline_id"] == "alt-a"
    assert data["baseline_variant_index"] == 2
    assert "generated_course" not in data
    assert "alt_baseline" not in data
    assert "baseline_variant" not in data
    assert "source_setup" not in data


def test_track_sampling_entry_metadata_views_are_absent_for_plain_entries() -> None:
    entry = TrackSamplingEntryConfig(id="mute_city")

    assert entry.baseline_variant_metadata() is None
    assert entry.alt_baseline_metadata() is None
    assert entry.generated_course_metadata() is None


def test_generated_course_entry_still_requires_seed_and_hash() -> None:
    with pytest.raises(ValidationError, match="generated_course_seed"):
        TrackSamplingEntryConfig(
            id="x_cup_slot_1",
            course_index=X_CUP_COURSE.course_index,
            mode=X_CUP_COURSE.race_mode,
            generated_course_kind=X_CUP_COURSE.generated_kind,
            generated_course_hash="abcd1234",
        )

    with pytest.raises(ValidationError, match="generated_course_hash"):
        TrackSamplingEntryConfig(
            id="x_cup_slot_1",
            course_index=X_CUP_COURSE.course_index,
            mode=X_CUP_COURSE.race_mode,
            generated_course_kind=X_CUP_COURSE.generated_kind,
            generated_course_seed=456,
        )
