# src/rl_fzerox/core/engine_tuning/state/objectives.py
"""Objective-specific projections for persisted engine-tuning observations."""

from __future__ import annotations

from dataclasses import replace

from rl_fzerox.core.engine_tuning.state.candidate import EngineTuningCandidateState
from rl_fzerox.core.engine_tuning.types import EngineTunerObjective


def project_candidate_for_objective(
    candidate: EngineTuningCandidateState,
    objective: EngineTunerObjective,
    *,
    safe_finish_rate_threshold: float = 0.9,
    prior_finish_time_seconds: float = 200.0,
) -> EngineTuningCandidateState | None:
    """Return a candidate with active scoring rebuilt for one tuner objective."""

    if objective == "finish_rate":
        return _finish_rate_projection(candidate)
    if objective == "safe_finish_time":
        return _safe_finish_time_projection(
            candidate,
            safe_finish_rate_threshold=safe_finish_rate_threshold,
            prior_finish_time_seconds=prior_finish_time_seconds,
        )
    return _finish_time_projection(candidate)


def _finish_rate_projection(
    candidate: EngineTuningCandidateState,
) -> EngineTuningCandidateState | None:
    if not candidate.has_valid_episode_statistics:
        return None
    finish_score_total = float(candidate.finish_count)
    return replace(
        candidate,
        score_count=candidate.episode_count,
        decayed_count=float(candidate.episode_count),
        decayed_score_total=finish_score_total,
        score_total=finish_score_total,
        best_score=1.0 if candidate.finish_count > 0 else 0.0,
    )


def _safe_finish_time_projection(
    candidate: EngineTuningCandidateState,
    *,
    safe_finish_rate_threshold: float,
    prior_finish_time_seconds: float,
) -> EngineTuningCandidateState | None:
    if not candidate.has_valid_episode_statistics:
        return None
    finish_rate = candidate.finish_rate_score or 0.0
    threshold = _clamp_unit_interval(safe_finish_rate_threshold)
    prior_penalty = max(1.0, float(prior_finish_time_seconds))
    if finish_rate >= threshold and candidate.finish_count > 0:
        mean_finish_score = candidate.finish_score_total / candidate.finish_count
        score_total = mean_finish_score * candidate.episode_count
        best_score = candidate.best_finish_score
    else:
        # Safe mode treats reliability as a hard gate. Below the gate, arms
        # compete by finish-rate gap instead of raw speed.
        gap = max(0.0, threshold - finish_rate)
        score_total = -prior_penalty * float(candidate.episode_count) * (1.0 + gap)
        best_score = -prior_penalty * (1.0 + gap)
    return replace(
        candidate,
        score_count=candidate.episode_count,
        decayed_count=float(candidate.episode_count),
        decayed_score_total=score_total,
        score_total=score_total,
        best_score=best_score,
    )


def _finish_time_projection(candidate: EngineTuningCandidateState) -> EngineTuningCandidateState:
    if candidate.finish_count <= 0:
        finish_score_total = 0.0
        best_finish_score = None
    else:
        finish_score_total = (
            candidate.finish_score_total
            if candidate.finish_score_total != 0.0 or candidate.best_finish_score is not None
            else candidate.score_total
        )
        best_finish_score = (
            candidate.best_finish_score
            if candidate.best_finish_score is not None
            else candidate.best_score
        )
    return replace(
        candidate,
        score_count=candidate.finish_count,
        decayed_count=float(candidate.finish_count),
        decayed_score_total=finish_score_total,
        score_total=finish_score_total,
        best_score=best_finish_score,
    )


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
