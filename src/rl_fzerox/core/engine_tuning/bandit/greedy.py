# src/rl_fzerox/core/engine_tuning/bandit/greedy.py
"""Greedy projection for the bandit engine tuner."""

from __future__ import annotations

from rl_fzerox.core.engine_tuning.bandit.projection import (
    _candidate_midpoint,
    _candidate_observation_count,
    _EngineEstimate,
    _EngineProjection,
)
from rl_fzerox.core.engine_tuning.types import (
    EngineTunerObjective,
    finish_time_ms_from_score,
)


def _bandit_greedy_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    safe_finish_rate_threshold: float,
) -> int:
    if not candidates:
        raise ValueError("adaptive engine tuning has no engine candidates")
    observed = tuple(
        engine_raw
        for engine_raw in candidates
        if _candidate_observation_count(
            projection.estimates[engine_raw],
            objective=projection.objective,
        )
        > 0
    )
    if not observed:
        midpoint = _candidate_midpoint(candidates)
        return min(candidates, key=lambda engine_raw: (abs(engine_raw - midpoint), engine_raw))
    midpoint = _candidate_midpoint(candidates)
    return max(
        observed,
        key=lambda engine_raw: _bandit_greedy_key(
            projection.estimates[engine_raw],
            engine_raw=engine_raw,
            midpoint=midpoint,
            objective=projection.objective,
            safe_finish_rate_threshold=safe_finish_rate_threshold,
        ),
    )


def _bandit_greedy_key(
    estimate: _EngineEstimate,
    *,
    engine_raw: int,
    midpoint: float,
    objective: EngineTunerObjective,
    safe_finish_rate_threshold: float,
) -> tuple[float, float, float, int, float, int]:
    if objective == "finish_rate":
        return (
            estimate.mean_score,
            estimate.best_score if estimate.best_score is not None else float("-inf"),
            estimate.finish_rate if estimate.finish_rate is not None else float("-inf"),
            estimate.exact_score_count,
            -abs(engine_raw - midpoint),
            -engine_raw,
        )
    if objective == "safe_finish_time":
        finish_rate = estimate.finish_rate if estimate.finish_rate is not None else 0.0
        is_safe = finish_rate >= safe_finish_rate_threshold and estimate.finish_count > 0
        best_finish_time_ms = (
            estimate.best_finish_time_ms
            if estimate.best_finish_time_ms is not None
            else 1_000_000_000
        )
        estimated_finish_time_ms = (
            finish_time_ms_from_score(estimate.mean_finish_score)
            if estimate.mean_finish_score is not None
            else 1_000_000_000
        )
        if is_safe:
            return (
                1.0,
                -estimated_finish_time_ms,
                -best_finish_time_ms,
                estimate.finish_count,
                -abs(engine_raw - midpoint),
                -engine_raw,
            )
        return (
            0.0,
            finish_rate,
            -estimated_finish_time_ms,
            estimate.episode_count,
            -abs(engine_raw - midpoint),
            -engine_raw,
        )
    best_finish_time_ms = (
        estimate.best_finish_time_ms if estimate.best_finish_time_ms is not None else 1_000_000_000
    )
    estimated_finish_time_ms = finish_time_ms_from_score(estimate.mean_score)
    return (
        -estimated_finish_time_ms,
        -best_finish_time_ms,
        estimate.mean_score,
        estimate.exact_score_count,
        -abs(engine_raw - midpoint),
        -engine_raw,
    )
