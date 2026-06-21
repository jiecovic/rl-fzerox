# tests/core/career_mode/test_attempt_timing.py
from __future__ import annotations

from rl_fzerox.core.career_mode.execution.attempt_timing import (
    CareerAttemptMenuJitter,
    career_attempt_menu_jitter_frames,
)


def test_career_attempt_menu_jitter_is_deterministic() -> None:
    assert (
        career_attempt_menu_jitter_frames(
            base_seed=1_528_865_290,
            attempt_id="attempt-1",
        )
        == 39
    )
    assert (
        career_attempt_menu_jitter_frames(
            base_seed=1_528_865_290,
            attempt_id="attempt-2",
        )
        == 54
    )


def test_career_attempt_menu_jitter_respects_configured_bound() -> None:
    jitter = CareerAttemptMenuJitter(max_neutral_frames=4)

    frames = jitter.frames_for(base_seed=1, attempt_id="attempt-a")

    assert 0 <= frames <= 4


def test_career_attempt_menu_jitter_requires_seed_and_attempt() -> None:
    jitter = CareerAttemptMenuJitter(max_neutral_frames=90)

    assert jitter.frames_for(base_seed=None, attempt_id="attempt-a") == 0
    assert jitter.frames_for(base_seed=1, attempt_id=None) == 0
    assert (
        CareerAttemptMenuJitter(max_neutral_frames=0).frames_for(
            base_seed=1,
            attempt_id="attempt-a",
        )
        == 0
    )
