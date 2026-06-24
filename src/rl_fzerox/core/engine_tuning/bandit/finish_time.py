# src/rl_fzerox/core/engine_tuning/bandit/finish_time.py
"""Finish-time sampling for the bandit engine tuner."""

from __future__ import annotations

from random import Random

from rl_fzerox.core.engine_tuning.bandit.projection import (
    _candidate_midpoint,
    _EngineProjection,
)


def _sample_finish_time_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
) -> tuple[int, float]:
    sampled_scores = _sample_scores(projection=projection, candidates=candidates, rng=rng)
    selected = max(
        candidates,
        key=lambda engine_raw: (
            sampled_scores[engine_raw],
            -abs(engine_raw - _candidate_midpoint(candidates)),
            -engine_raw,
        ),
    )
    return selected, sampled_scores[selected]


def _sample_scores(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
) -> dict[int, float]:
    return {
        engine_raw: projection.estimates[engine_raw].mean_score
        + projection.estimates[engine_raw].uncertainty_score * rng.gauss(0.0, 1.0)
        for engine_raw in candidates
    }
