# src/rl_fzerox/core/engine_tuning/bandit_sampling.py
"""Sampling and greedy projection helpers for the bandit engine tuner."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random

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


def _sample_finish_rate(estimate: _EngineEstimate, *, rng: Random) -> float:
    failures = max(0, estimate.episode_count - estimate.finish_count)
    return rng.betavariate(1.0 + estimate.finish_count, 1.0 + failures)


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


def _beta_finish_rate_std(finish_count: int, episode_count: int) -> float:
    failures = max(0, int(episode_count) - int(finish_count))
    alpha = 1.0 + max(0, int(finish_count))
    beta = 1.0 + failures
    total = alpha + beta
    return sqrt((alpha * beta) / (total * total * (total + 1.0)))


def _candidate_midpoint(candidates: tuple[int, ...]) -> float:
    return (candidates[0] + candidates[-1]) / 2.0


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
