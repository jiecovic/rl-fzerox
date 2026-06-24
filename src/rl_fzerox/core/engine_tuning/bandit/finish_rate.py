# src/rl_fzerox/core/engine_tuning/bandit/finish_rate.py
"""Finish-rate sampling for the bandit engine tuner."""

from __future__ import annotations

from random import Random

from rl_fzerox.core.engine_tuning.bandit.projection import (
    _candidate_midpoint,
    _EngineEstimate,
    _EngineProjection,
)


def _sample_finish_rate_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
) -> tuple[int, float]:
    sampled_rates = {
        engine_raw: _sample_finish_rate(projection.estimates[engine_raw], rng=rng)
        for engine_raw in candidates
    }
    midpoint = _candidate_midpoint(candidates)
    selected = max(
        candidates,
        key=lambda engine_raw: (
            sampled_rates[engine_raw],
            projection.estimates[engine_raw].finish_rate
            if projection.estimates[engine_raw].finish_rate is not None
            else float("-inf"),
            projection.estimates[engine_raw].finish_count,
            projection.estimates[engine_raw].episode_count,
            -abs(engine_raw - midpoint),
            -engine_raw,
        ),
    )
    return selected, sampled_rates[selected]


def _sample_finish_rate(estimate: _EngineEstimate, *, rng: Random) -> float:
    failures = max(0, estimate.episode_count - estimate.finish_count)
    return rng.betavariate(1.0 + estimate.finish_count, 1.0 + failures)
