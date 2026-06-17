# src/rl_fzerox/core/engine_tuning/bandit.py
"""Aggregate bandit backend for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningCandidateState,
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.types import (
    BanditEngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    engine_bucket_candidates,
    finish_time_ms_from_score,
    finish_time_score,
    successful_finish_time_ms,
)


class BanditEngineTuner:
    """Choose from coarse engine buckets using measured finish aggregates."""

    def __init__(
        self,
        *,
        settings: BanditEngineTunerSettings,
        state: EngineTuningRuntimeState | None = None,
    ) -> None:
        self._settings = settings
        self._state = _bandit_state_or_empty(state, settings=settings)

    @property
    def state(self) -> EngineTuningRuntimeState:
        return self._state

    def choose(self, context: EngineTuningContext, *, seed: int | None) -> EngineTuningChoice:
        """Sample one bucketed integer engine value for the given context."""

        rng = Random(seed) if seed is not None else Random()
        candidates = self._candidates()
        if rng.random() < max(0.0, min(1.0, self._settings.uniform_exploration)):
            selected = rng.choice(candidates)
            estimate = self._context_projection(context, candidates).estimates[selected]
            return self._choice_for(context, selected, estimate=estimate, sampled_score=None)

        projection = self._context_projection(context, candidates)
        unobserved = tuple(
            engine_raw
            for engine_raw in candidates
            if projection.estimates[engine_raw].exact_finish_count <= 0
        )
        if unobserved:
            selected = rng.choice(unobserved)
            estimate = projection.estimates[selected]
            return self._choice_for(context, selected, estimate=estimate, sampled_score=None)

        sampled_scores = _sample_scores(projection=projection, candidates=candidates, rng=rng)
        selected = max(
            candidates,
            key=lambda engine_raw: (
                sampled_scores[engine_raw],
                -abs(engine_raw - _candidate_midpoint(candidates)),
                -engine_raw,
            ),
        )
        return self._choice_for(
            context,
            selected,
            estimate=projection.estimates[selected],
            sampled_score=sampled_scores[selected],
        )

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the best measured bucket without random exploration."""

        candidates = self._candidates()
        projection = self._context_projection(context, candidates)
        engine_raw = _bandit_greedy_engine_setting(projection=projection, candidates=candidates)
        return self._choice_for(
            context,
            engine_raw,
            estimate=projection.estimates[engine_raw],
            sampled_score=None,
        )

    def distribution(
        self,
        context: EngineTuningContext,
        *,
        seed: int,
        draws: int = 512,
    ) -> tuple[EngineTuningCandidateEstimate, ...]:
        """Estimate the current stochastic reset distribution for one context."""

        candidates = self._candidates()
        projection = self._context_projection(context, candidates)
        probabilities = _selection_probabilities(
            projection=projection,
            candidates=candidates,
            uniform_exploration=self._settings.uniform_exploration,
            seed=seed,
            draws=draws,
        )
        return tuple(
            _candidate_estimate(
                projection.estimates[engine_raw],
                engine_raw,
                probability=probabilities[engine_raw],
            )
            for engine_raw in candidates
        )

    def record(self, outcome: EngineTuningEpisodeOutcome) -> EngineTuningRuntimeState:
        """Update the state from one terminal episode result."""

        return self.record_many((outcome,))

    def record_many(
        self,
        outcomes: tuple[EngineTuningEpisodeOutcome, ...],
    ) -> EngineTuningRuntimeState:
        """Update aggregate observations once from one rollout batch."""

        successful = tuple(_successful_score(outcome) for outcome in outcomes)
        successful = tuple(item for item in successful if item is not None)
        if not successful:
            return self._state
        next_state = self._state
        candidates = self._candidates()
        changed = False
        for outcome, score, finish_time_ms in successful:
            bucket = _exact_bucket(outcome.engine_setting_raw_value, candidates)
            if bucket is None:
                continue
            candidate = _candidate_from_state(
                next_state,
                outcome.context,
                bucket,
            ).record(score=score, finish_time_ms=finish_time_ms)
            next_state = next_state.with_candidate(candidate)
            changed = True
        if not changed:
            return self._state
        self._state = _canonical_bandit_state(
            next_state.with_model_state(None),
            settings=self._settings,
        )
        return self._state

    def score(self, outcome: EngineTuningEpisodeOutcome) -> float:
        """Return a higher-is-better negative finish-time score."""

        finish_time_ms = successful_finish_time_ms(outcome)
        if finish_time_ms is None:
            return self._prior_score()
        return finish_time_score(finish_time_ms)

    def _choice_for(
        self,
        context: EngineTuningContext,
        engine_raw: int,
        *,
        estimate: _EngineEstimate,
        sampled_score: float | None,
    ) -> EngineTuningChoice:
        return EngineTuningChoice(
            context=context,
            engine_setting_raw_value=engine_raw,
            sampled_score=estimate.mean_score if sampled_score is None else sampled_score,
            mean_score=estimate.mean_score,
            finish_count=estimate.exact_finish_count,
            estimated_finish_time_ms=finish_time_ms_from_score(estimate.mean_score),
            best_finish_time_ms=estimate.best_finish_time_ms,
        )

    def _context_projection(
        self,
        context: EngineTuningContext,
        candidates: tuple[int, ...],
    ) -> _EngineProjection:
        bucket_candidates = _bucket_candidates(
            self._state,
            context=context,
            candidates=candidates,
        )
        estimates: dict[int, _EngineEstimate] = {}
        for engine_raw in candidates:
            candidate = bucket_candidates.get(engine_raw)
            estimates[engine_raw] = self._candidate_estimate(candidate)
        return _EngineProjection(estimates=estimates)

    def _candidate_estimate(
        self,
        candidate: EngineTuningCandidateState | None,
    ) -> _EngineEstimate:
        if candidate is None or candidate.mean_score is None or candidate.finish_count <= 0:
            return _EngineEstimate(
                mean_score=self._prior_score(),
                uncertainty_score=max(0.0, float(self._settings.exploration_seconds)),
                exact_finish_count=0,
                best_finish_time_ms=None,
            )
        return _EngineEstimate(
            mean_score=candidate.mean_score,
            uncertainty_score=_candidate_uncertainty(
                candidate,
                exploration_seconds=self._settings.exploration_seconds,
            ),
            exact_finish_count=candidate.finish_count,
            best_finish_time_ms=candidate.best_time_ms,
        )

    def _candidates(self) -> tuple[int, ...]:
        return engine_bucket_candidates(
            minimum=self._settings.min_raw_value,
            maximum=self._settings.max_raw_value,
            slider_spacing=self._settings.slider_spacing,
        )

    def _prior_score(self) -> float:
        return -max(1.0, float(self._settings.prior_finish_time_seconds))


@dataclass(frozen=True, slots=True)
class _EngineEstimate:
    mean_score: float
    uncertainty_score: float
    exact_finish_count: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class _EngineProjection:
    estimates: dict[int, _EngineEstimate]


def _bucket_candidates(
    state: EngineTuningRuntimeState,
    *,
    context: EngineTuningContext,
    candidates: tuple[int, ...],
) -> dict[int, EngineTuningCandidateState]:
    buckets: dict[int, EngineTuningCandidateState] = {}
    for candidate in state.candidates:
        if candidate.context_key != context.key or candidate.finish_count <= 0:
            continue
        bucket = _exact_bucket(candidate.engine_setting_raw_value, candidates)
        if bucket is None:
            continue
        buckets[bucket] = _bucketed_candidate(
            existing=buckets.get(bucket),
            source=candidate,
            engine_raw=bucket,
        )
    return buckets


def _exact_bucket(raw_value: int, candidates: tuple[int, ...]) -> int | None:
    raw_bucket = int(raw_value)
    if raw_bucket in candidates:
        return raw_bucket
    return None


def _canonical_bandit_state(
    state: EngineTuningRuntimeState,
    *,
    settings: BanditEngineTunerSettings,
) -> EngineTuningRuntimeState:
    candidates = engine_bucket_candidates(
        minimum=settings.min_raw_value,
        maximum=settings.max_raw_value,
        slider_spacing=settings.slider_spacing,
    )
    buckets: dict[tuple[str, int], EngineTuningCandidateState] = {}
    for candidate in state.candidates:
        if candidate.finish_count <= 0:
            continue
        bucket = _exact_bucket(candidate.engine_setting_raw_value, candidates)
        if bucket is None:
            continue
        key = (candidate.context_key, bucket)
        buckets[key] = _bucketed_candidate(
            existing=buckets.get(key),
            source=candidate,
            engine_raw=bucket,
        )
    return EngineTuningRuntimeState(
        version=state.version,
        update_count=state.update_count,
        candidates=tuple(
            sorted(
                buckets.values(),
                key=lambda candidate: (
                    candidate.context_key,
                    candidate.engine_setting_raw_value,
                ),
            )
        ),
        model_state=None,
    )


def _bucketed_candidate(
    *,
    existing: EngineTuningCandidateState | None,
    source: EngineTuningCandidateState,
    engine_raw: int,
) -> EngineTuningCandidateState:
    if existing is None:
        return EngineTuningCandidateState(
            context_key=source.context_key,
            course_key=source.course_key,
            vehicle_id=source.vehicle_id,
            engine_setting_raw_value=int(engine_raw),
            finish_count=max(0, int(source.finish_count)),
            decayed_count=max(0.0, float(source.decayed_count)),
            decayed_score_total=float(source.decayed_score_total),
            score_total=float(source.score_total),
            best_score=source.best_score,
            best_time_ms=source.best_time_ms,
        )
    return EngineTuningCandidateState(
        context_key=existing.context_key,
        course_key=existing.course_key,
        vehicle_id=existing.vehicle_id,
        engine_setting_raw_value=existing.engine_setting_raw_value,
        finish_count=existing.finish_count + max(0, int(source.finish_count)),
        decayed_count=existing.decayed_count + max(0.0, float(source.decayed_count)),
        decayed_score_total=existing.decayed_score_total + float(source.decayed_score_total),
        score_total=existing.score_total + float(source.score_total),
        best_score=_max_optional_score(existing.best_score, source.best_score),
        best_time_ms=_min_optional_time(existing.best_time_ms, source.best_time_ms),
    )


def _max_optional_score(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_optional_time(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _candidate_from_state(
    state: EngineTuningRuntimeState,
    context: EngineTuningContext,
    engine_raw: int,
) -> EngineTuningCandidateState:
    candidate = state.candidate_map().get((context.key, int(engine_raw)))
    if candidate is not None:
        return candidate
    return EngineTuningCandidateState(
        context_key=context.key,
        course_key=context.course_key,
        vehicle_id=context.vehicle_id,
        engine_setting_raw_value=int(engine_raw),
    )


def _successful_score(
    outcome: EngineTuningEpisodeOutcome,
) -> tuple[EngineTuningEpisodeOutcome, float, int] | None:
    finish_time_ms = successful_finish_time_ms(outcome)
    if finish_time_ms is None:
        return None
    return outcome, finish_time_score(finish_time_ms), finish_time_ms


def _selection_probabilities(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    uniform_exploration: float,
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
    seed: int,
    draws: int,
) -> dict[int, float]:
    unobserved = tuple(
        engine_raw
        for engine_raw in candidates
        if projection.estimates[engine_raw].exact_finish_count <= 0
    )
    if unobserved:
        return {
            engine_raw: (1.0 / len(unobserved) if engine_raw in unobserved else 0.0)
            for engine_raw in candidates
        }

    rng = Random(seed)
    draw_count = max(1, int(draws))
    counts = dict.fromkeys(candidates, 0)
    for _ in range(draw_count):
        sampled_scores = _sample_scores(projection=projection, candidates=candidates, rng=rng)
        selected = max(
            candidates,
            key=lambda engine_raw: (
                sampled_scores[engine_raw],
                -abs(engine_raw - _candidate_midpoint(candidates)),
                -engine_raw,
            ),
        )
        counts[selected] += 1
    return {engine_raw: counts[engine_raw] / draw_count for engine_raw in candidates}


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
        finish_count=estimate.exact_finish_count,
        best_finish_time_ms=estimate.best_finish_time_ms,
    )


def _candidate_uncertainty(
    candidate: EngineTuningCandidateState,
    *,
    exploration_seconds: float,
) -> float:
    return max(0.25, float(exploration_seconds)) / max(1.0, candidate.decayed_count) ** 0.5


def _candidate_midpoint(candidates: tuple[int, ...]) -> float:
    return (candidates[0] + candidates[-1]) / 2.0


def _bandit_greedy_engine_setting(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
) -> int:
    if not candidates:
        raise ValueError("adaptive engine tuning has no engine candidates")
    observed = tuple(
        engine_raw
        for engine_raw in candidates
        if projection.estimates[engine_raw].exact_finish_count > 0
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
        ),
    )


def _bandit_greedy_key(
    estimate: _EngineEstimate,
    *,
    engine_raw: int,
    midpoint: float,
) -> tuple[float, int, int, float, int]:
    best_finish_time_ms = (
        estimate.best_finish_time_ms if estimate.best_finish_time_ms is not None else 1_000_000_000
    )
    return (
        estimate.mean_score,
        estimate.exact_finish_count,
        -best_finish_time_ms,
        -abs(engine_raw - midpoint),
        -engine_raw,
    )


def _bandit_state_or_empty(
    state: EngineTuningRuntimeState | None,
    *,
    settings: BanditEngineTunerSettings,
) -> EngineTuningRuntimeState:
    if state is None:
        return empty_engine_tuning_state()
    return _canonical_bandit_state(state, settings=settings)
