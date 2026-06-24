# src/rl_fzerox/core/engine_tuning/bandit/sampling.py
"""Sampling orchestration for the bandit engine tuner."""

from __future__ import annotations

from random import Random

from rl_fzerox.core.engine_tuning.bandit.finish_rate import (
    _sample_finish_rate_engine_setting,
)
from rl_fzerox.core.engine_tuning.bandit.finish_time import (
    _sample_finish_time_engine_setting,
)
from rl_fzerox.core.engine_tuning.bandit.projection import (
    _candidate_observation_count,
    _EngineEstimate,
    _EngineProjection,
)
from rl_fzerox.core.engine_tuning.bandit.safe_finish_time import (
    _sample_safe_finish_time_engine_setting,
)
from rl_fzerox.core.engine_tuning.types import EngineTunerObjective


def _selection_probabilities(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    uniform_exploration: float,
    safe_finish_rate_threshold: float,
    prior_finish_time_seconds: float,
    min_finish_rate_observations: int,
    seed: int,
    draws: int,
) -> dict[int, float]:
    if not candidates:
        raise ValueError("adaptive engine tuning has no engine candidates")
    uniform_probability = 1.0 / len(candidates)
    exploration = max(0.0, min(1.0, float(uniform_exploration)))
    model_probabilities = _model_probabilities(
        projection=projection,
        candidates=candidates,
        safe_finish_rate_threshold=safe_finish_rate_threshold,
        prior_finish_time_seconds=prior_finish_time_seconds,
        min_finish_rate_observations=min_finish_rate_observations,
        seed=seed,
        draws=draws,
    )
    return {
        engine_raw: exploration * uniform_probability
        + (1.0 - exploration) * model_probabilities[engine_raw]
        for engine_raw in candidates
    }


def _model_probabilities(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    safe_finish_rate_threshold: float,
    prior_finish_time_seconds: float,
    min_finish_rate_observations: int,
    seed: int,
    draws: int,
) -> dict[int, float]:
    warmup_candidates = _warmup_candidates(
        projection=projection,
        candidates=candidates,
        min_finish_rate_observations=min_finish_rate_observations,
    )
    if warmup_candidates:
        return {
            engine_raw: (1.0 / len(warmup_candidates) if engine_raw in warmup_candidates else 0.0)
            for engine_raw in candidates
        }

    rng = Random(seed)
    draw_count = max(1, int(draws))
    counts = dict.fromkeys(candidates, 0)
    for _ in range(draw_count):
        selected, _sampled_score = _sample_engine_setting(
            projection=projection,
            candidates=candidates,
            rng=rng,
            threshold=safe_finish_rate_threshold,
            prior_finish_time_seconds=prior_finish_time_seconds,
        )
        counts[selected] += 1
    return {engine_raw: counts[engine_raw] / draw_count for engine_raw in candidates}


def _warmup_candidates(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    min_finish_rate_observations: int,
) -> tuple[int, ...]:
    minimum = max(0, int(min_finish_rate_observations))
    return tuple(
        engine_raw
        for engine_raw in candidates
        if _candidate_needs_warmup(
            projection.estimates[engine_raw],
            objective=projection.objective,
            min_finish_rate_observations=minimum,
        )
    )


def _candidate_needs_warmup(
    estimate: _EngineEstimate,
    *,
    objective: EngineTunerObjective,
    min_finish_rate_observations: int,
) -> bool:
    if objective in {"finish_rate", "safe_finish_time"}:
        return estimate.episode_count < max(0, int(min_finish_rate_observations))
    return _candidate_observation_count(estimate, objective=objective) <= 0


def _sample_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    rng: Random,
    threshold: float = 0.9,
    prior_finish_time_seconds: float = 200.0,
) -> tuple[int, float]:
    if projection.objective == "safe_finish_time":
        return _sample_safe_finish_time_engine_setting(
            projection=projection,
            candidates=candidates,
            rng=rng,
            threshold=threshold,
            prior_finish_time_seconds=prior_finish_time_seconds,
        )
    if projection.objective == "finish_rate":
        return _sample_finish_rate_engine_setting(
            projection=projection,
            candidates=candidates,
            rng=rng,
        )
    return _sample_finish_time_engine_setting(
        projection=projection,
        candidates=candidates,
        rng=rng,
    )
