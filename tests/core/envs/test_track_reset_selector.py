# tests/core/envs/test_track_reset_selector.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.envs.engine.reset import select_reset_track_by_course_id
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig


def test_select_reset_track_by_course_id_uses_matching_entry() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="balanced",
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                course_name="Mute City",
                baseline_state_path=Path("mute_city.state"),
            ),
            TrackSamplingEntryConfig(
                id="silence",
                course_id="silence",
                course_name="Silence",
                baseline_state_path=Path("silence.state"),
            ),
        ),
    )

    selected = select_reset_track_by_course_id(config, course_id="silence")

    assert selected is not None
    assert selected.id == "silence"
    assert selected.course_id == "silence"
    assert selected.sampling_mode == "locked"


def test_select_reset_track_by_course_id_returns_none_without_match() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                baseline_state_path=Path("mute_city.state"),
            ),
        ),
    )

    assert select_reset_track_by_course_id(config, course_id="silence") is None
