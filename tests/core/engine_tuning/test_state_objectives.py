# tests/core/engine_tuning/test_state_objectives.py
from __future__ import annotations

import pytest

from rl_fzerox.core.engine_tuning.state import EngineTuningCandidateState
from rl_fzerox.core.engine_tuning.state.objectives import project_candidate_for_objective


def test_finish_time_projection_uses_finish_time_aggregates() -> None:
    candidate = _candidate(
        score_count=10,
        finish_count=2,
        decayed_count=10.0,
        decayed_score_total=123.0,
        score_total=123.0,
        best_score=100.0,
        finish_score_total=-150.0,
        best_finish_score=-70.0,
    )

    projected = project_candidate_for_objective(candidate, "finish_time")

    assert projected is not None
    assert projected.score_count == 2
    assert projected.decayed_count == 2.0
    assert projected.decayed_score_total == -150.0
    assert projected.score_total == -150.0
    assert projected.best_score == -70.0
    assert candidate.score_count == 10


def test_finish_rate_projection_uses_episode_finish_counts() -> None:
    candidate = _candidate(
        episode_count=10,
        finish_count=6,
        completion_score_total=8.0,
        score_count=2,
        score_total=-150.0,
        best_score=-70.0,
    )

    projected = project_candidate_for_objective(candidate, "finish_rate")

    assert projected is not None
    assert projected.score_count == 10
    assert projected.decayed_count == 10.0
    assert projected.decayed_score_total == 6.0
    assert projected.score_total == 6.0
    assert projected.best_score == 1.0


def test_safe_finish_time_projection_uses_finish_time_after_gate() -> None:
    candidate = _candidate(
        episode_count=10,
        finish_count=8,
        completion_score_total=9.0,
        finish_score_total=-560.0,
        best_finish_score=-66.0,
    )

    projected = project_candidate_for_objective(
        candidate,
        "safe_finish_time",
        safe_finish_rate_threshold=0.75,
        prior_finish_time_seconds=100.0,
    )

    assert projected is not None
    assert projected.score_count == 10
    assert projected.decayed_count == 10.0
    assert projected.decayed_score_total == -700.0
    assert projected.score_total == -700.0
    assert projected.best_score == -66.0


def test_safe_finish_time_projection_penalizes_candidates_below_gate() -> None:
    candidate = _candidate(
        episode_count=10,
        finish_count=4,
        completion_score_total=8.0,
        finish_score_total=-280.0,
        best_finish_score=-65.0,
    )

    projected = project_candidate_for_objective(
        candidate,
        "safe_finish_time",
        safe_finish_rate_threshold=0.75,
        prior_finish_time_seconds=100.0,
    )

    assert projected is not None
    assert projected.score_count == 10
    assert projected.decayed_count == 10.0
    assert projected.decayed_score_total == pytest.approx(-1350.0)
    assert projected.score_total == pytest.approx(-1350.0)
    assert projected.best_score == pytest.approx(-135.0)


def test_episode_rate_projections_reject_invalid_episode_statistics() -> None:
    candidate = _candidate(
        episode_count=1,
        finish_count=3,
        completion_score_total=5.0,
    )

    assert project_candidate_for_objective(candidate, "finish_rate") is None
    assert project_candidate_for_objective(candidate, "safe_finish_time") is None
    assert project_candidate_for_objective(candidate, "finish_time") is not None


def _candidate(**overrides: object) -> EngineTuningCandidateState:
    defaults = {
        "context_key": "mute_city|blue_falcon",
        "course_key": "mute_city",
        "vehicle_id": "blue_falcon",
        "engine_setting_raw_value": 70,
    }
    defaults.update(overrides)
    return EngineTuningCandidateState(**defaults)
