# src/rl_fzerox/core/engine_tuning/bandit/safe_finish_time.py
"""Safe-finish-time sampling for the bandit engine tuner."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from rl_fzerox.core.engine_tuning.bandit.finish_rate import _sample_finish_rate
from rl_fzerox.core.engine_tuning.bandit.projection import (
    _candidate_midpoint,
    _EngineEstimate,
    _EngineProjection,
)


def _sample_safe_finish_time_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
    threshold: float,
    prior_finish_time_seconds: float,
) -> tuple[int, float]:
    sampled = {
        engine_raw: _sample_safe_finish_time_candidate(
            projection.estimates[engine_raw],
            rng=rng,
            prior_finish_time_seconds=prior_finish_time_seconds,
        )
        for engine_raw in candidates
    }
    safe_candidates = tuple(
        engine_raw for engine_raw in candidates if sampled[engine_raw].finish_rate >= threshold
    )
    midpoint = _candidate_midpoint(candidates)
    if safe_candidates:
        selected = max(
            safe_candidates,
            key=lambda engine_raw: _safe_finish_time_selection_key(
                projection.estimates[engine_raw],
                sampled[engine_raw],
                engine_raw=engine_raw,
                midpoint=midpoint,
            ),
        )
        return selected, sampled[selected].finish_time_score
    selected = max(
        candidates,
        key=lambda engine_raw: (
            sampled[engine_raw].finish_rate,
            sampled[engine_raw].finish_time_score,
            projection.estimates[engine_raw].episode_count,
            -abs(engine_raw - midpoint),
            -engine_raw,
        ),
    )
    return selected, sampled[selected].finish_rate


@dataclass(frozen=True, slots=True)
class _SafeFinishTimeSample:
    finish_rate: float
    finish_time_score: float


def _sample_safe_finish_time_candidate(
    estimate: _EngineEstimate,
    *,
    rng: Random,
    prior_finish_time_seconds: float,
) -> _SafeFinishTimeSample:
    finish_rate = _sample_finish_rate(estimate, rng=rng)
    mean_finish_score = (
        estimate.mean_finish_score
        if estimate.mean_finish_score is not None
        else -max(1.0, float(prior_finish_time_seconds))
    )
    finish_time_score = mean_finish_score + estimate.uncertainty_score * rng.gauss(0.0, 1.0)
    return _SafeFinishTimeSample(
        finish_rate=finish_rate,
        finish_time_score=finish_time_score,
    )


def _safe_finish_time_selection_key(
    estimate: _EngineEstimate,
    sample: _SafeFinishTimeSample,
    *,
    engine_raw: int,
    midpoint: float,
) -> tuple[bool, float, float, int, float, int]:
    if estimate.finish_count <= 0:
        # The finish-time model only has a real timing observation after a
        # successful finish. Once the sampled safety posterior clears the
        # constraint, collect that first timing observation before ranking the
        # arm against measured finish times.
        return (
            True,
            sample.finish_rate,
            -float(estimate.episode_count),
            estimate.finish_count,
            -abs(engine_raw - midpoint),
            -engine_raw,
        )
    return (
        False,
        sample.finish_time_score,
        sample.finish_rate,
        estimate.finish_count,
        -abs(engine_raw - midpoint),
        -engine_raw,
    )
