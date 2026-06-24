# src/rl_fzerox/core/engine_tuning/bandit/projection.py
"""Bandit projection and estimate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from rl_fzerox.core.engine_tuning.state import EngineTuningCandidateState
from rl_fzerox.core.engine_tuning.types import (
    EngineTunerObjective,
    EngineTuningCandidateEstimate,
    finish_time_ms_from_score,
)


@dataclass(frozen=True, slots=True)
class _EngineEstimate:
    mean_score: float
    uncertainty_score: float
    exact_score_count: int
    episode_count: int
    finish_count: int
    finish_rate: float | None
    mean_finish_score: float | None
    best_finish_time_ms: int | None
    best_score: float | None


@dataclass(frozen=True, slots=True)
class _EngineProjection:
    objective: EngineTunerObjective
    estimates: dict[int, _EngineEstimate]


def _candidate_estimate(
    estimate: _EngineEstimate,
    engine_raw: int,
    *,
    probability: float,
) -> EngineTuningCandidateEstimate:
    return EngineTuningCandidateEstimate(
        engine_setting_raw_value=engine_raw,
        probability=probability,
        mean_score=estimate.mean_score,
        uncertainty_score=estimate.uncertainty_score,
        estimated_finish_time_ms=finish_time_ms_from_score(estimate.mean_score),
        score_count=estimate.exact_score_count,
        finish_count=estimate.finish_count,
        best_finish_time_ms=estimate.best_finish_time_ms,
        best_score=estimate.best_score,
    )


def _candidate_observation_count(
    estimate: _EngineEstimate,
    *,
    objective: EngineTunerObjective,
) -> int:
    if objective in {"safe_finish_time", "finish_rate"}:
        return estimate.episode_count
    return estimate.exact_score_count


def _candidate_uncertainty(
    candidate: EngineTuningCandidateState,
    *,
    objective: EngineTunerObjective,
    exploration_seconds: float,
) -> float:
    if objective == "finish_rate":
        return _beta_finish_rate_std(candidate.finish_count, candidate.episode_count)
    if objective == "safe_finish_time":
        return max(0.25, float(exploration_seconds)) / max(1.0, candidate.finish_count) ** 0.5
    return max(0.25, float(exploration_seconds)) / max(1.0, candidate.decayed_count) ** 0.5


def _candidate_midpoint(candidates: tuple[int, ...]) -> float:
    return (candidates[0] + candidates[-1]) / 2.0


def _beta_finish_rate_std(finish_count: int, episode_count: int) -> float:
    failures = max(0, int(episode_count) - int(finish_count))
    alpha = 1.0 + max(0, int(finish_count))
    beta = 1.0 + failures
    total = alpha + beta
    return sqrt((alpha * beta) / (total * total * (total + 1.0)))
