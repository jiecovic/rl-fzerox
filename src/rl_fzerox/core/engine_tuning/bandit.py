# src/rl_fzerox/core/engine_tuning/bandit.py
"""Aggregate bandit backend for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from rl_fzerox.core.engine_tuning.state import (
    EngineTuningCandidateState,
    EngineTuningRuntimeState,
    empty_engine_tuning_state_for,
    engine_tuning_state_with_objective,
)
from rl_fzerox.core.engine_tuning.types import (
    BanditEngineTunerSettings,
    EngineTunerObjective,
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
            if _candidate_observation_count(
                projection.estimates[engine_raw],
                objective=projection.objective,
            )
            <= 0
        )
        if unobserved:
            selected = rng.choice(unobserved)
            estimate = projection.estimates[selected]
            return self._choice_for(context, selected, estimate=estimate, sampled_score=None)

        selected, sampled_score = _sample_engine_setting(
            projection=projection,
            candidates=candidates,
            rng=rng,
            threshold=self._settings.safe_finish_rate_threshold,
            prior_finish_time_seconds=self._settings.prior_finish_time_seconds,
        )
        return self._choice_for(
            context,
            selected,
            estimate=projection.estimates[selected],
            sampled_score=sampled_score,
        )

    def recommendation(self, context: EngineTuningContext) -> EngineTuningChoice:
        """Return the best measured bucket without random exploration."""

        candidates = self._candidates()
        projection = self._context_projection(context, candidates)
        engine_raw = _bandit_greedy_engine_setting(
            projection=projection,
            candidates=candidates,
            safe_finish_rate_threshold=self._settings.safe_finish_rate_threshold,
        )
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
            safe_finish_rate_threshold=self._settings.safe_finish_rate_threshold,
            prior_finish_time_seconds=self._settings.prior_finish_time_seconds,
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

        if not outcomes:
            return self._state
        next_state = self._state
        candidates = self._candidates()
        changed = False
        for outcome in outcomes:
            bucket = _exact_bucket(outcome.engine_setting_raw_value, candidates)
            if bucket is None:
                continue
            score, finish_time_ms = _objective_score(outcome, settings=self._settings)
            candidate = _candidate_from_state(
                next_state,
                outcome.context,
                bucket,
            ).record(
                score=score,
                completion_fraction=outcome.completion_fraction,
                finish_time_ms=finish_time_ms,
                episode_return=outcome.episode_return,
            )
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
        """Return the configured higher-is-better objective score."""

        score, _finish_time_ms = _objective_score(outcome, settings=self._settings)
        return self._prior_score() if score is None else score

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
            score_count=estimate.exact_score_count,
            finish_count=estimate.finish_count,
            estimated_finish_time_ms=finish_time_ms_from_score(estimate.mean_score),
            best_finish_time_ms=estimate.best_finish_time_ms,
            best_score=estimate.best_score,
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
        return _EngineProjection(objective=self._settings.objective, estimates=estimates)

    def _candidate_estimate(
        self,
        candidate: EngineTuningCandidateState | None,
    ) -> _EngineEstimate:
        if candidate is None:
            return _EngineEstimate(
                mean_score=self._prior_score(),
                uncertainty_score=max(0.0, float(self._settings.exploration_seconds)),
                exact_score_count=0,
                episode_count=0,
                finish_count=0,
                finish_rate=None,
                mean_finish_score=None,
                best_finish_time_ms=None,
                best_score=None,
            )
        if candidate.mean_score is None or candidate.active_score_count <= 0:
            return _EngineEstimate(
                mean_score=self._prior_score(),
                uncertainty_score=max(0.0, float(self._settings.exploration_seconds)),
                exact_score_count=0,
                episode_count=candidate.episode_count,
                finish_count=candidate.finish_count,
                finish_rate=candidate.finish_rate_score,
                mean_finish_score=candidate.mean_finish_score,
                best_finish_time_ms=candidate.best_time_ms,
                best_score=candidate.best_score,
            )
        finish_rate = candidate.finish_rate_score
        return _EngineEstimate(
            mean_score=candidate.mean_score,
            uncertainty_score=_candidate_uncertainty(
                candidate,
                exploration_seconds=self._settings.exploration_seconds,
            ),
            exact_score_count=candidate.active_score_count,
            episode_count=candidate.episode_count,
            finish_count=candidate.finish_count,
            finish_rate=finish_rate,
            mean_finish_score=candidate.mean_finish_score,
            best_finish_time_ms=candidate.best_time_ms,
            best_score=candidate.best_score,
        )

    def _candidates(self) -> tuple[int, ...]:
        return engine_bucket_candidates(bucket_raw_values=self._settings.bucket_raw_values)

    def _prior_score(self) -> float:
        if self._settings.objective == "finish_rate":
            return 0.0
        return -max(1.0, float(self._settings.prior_finish_time_seconds))


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


def _bucket_candidates(
    state: EngineTuningRuntimeState,
    *,
    context: EngineTuningContext,
    candidates: tuple[int, ...],
) -> dict[int, EngineTuningCandidateState]:
    buckets: dict[int, EngineTuningCandidateState] = {}
    for candidate in state.candidates:
        if candidate.context_key != context.key or candidate.observation_count <= 0:
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
    state = engine_tuning_state_with_objective(
        state,
        objective=settings.objective,
        reward_fingerprint=settings.reward_fingerprint,
        safe_finish_rate_threshold=settings.safe_finish_rate_threshold,
        prior_finish_time_seconds=settings.prior_finish_time_seconds,
    )
    candidates = engine_bucket_candidates(bucket_raw_values=settings.bucket_raw_values)
    buckets: dict[tuple[str, int], EngineTuningCandidateState] = {}
    for candidate in state.candidates:
        if candidate.observation_count <= 0:
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
        objective=state.objective,
        reward_fingerprint=state.reward_fingerprint,
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
            score_count=max(0, int(source.active_score_count)),
            episode_count=max(0, int(source.episode_count)),
            finish_count=max(0, int(source.finish_count)),
            return_count=max(0, int(source.return_count)),
            decayed_count=max(0.0, float(source.decayed_count)),
            decayed_score_total=float(source.decayed_score_total),
            score_total=float(source.score_total),
            best_score=source.best_score,
            completion_score_total=float(source.completion_score_total),
            best_completion_score=source.best_completion_score,
            finish_score_total=float(source.finish_score_total),
            best_finish_score=source.best_finish_score,
            return_score_total=float(source.return_score_total),
            best_return_score=source.best_return_score,
            best_time_ms=source.best_time_ms,
        )
    return EngineTuningCandidateState(
        context_key=existing.context_key,
        course_key=existing.course_key,
        vehicle_id=existing.vehicle_id,
        engine_setting_raw_value=existing.engine_setting_raw_value,
        score_count=existing.active_score_count + max(0, int(source.active_score_count)),
        episode_count=existing.episode_count + max(0, int(source.episode_count)),
        finish_count=existing.finish_count + max(0, int(source.finish_count)),
        return_count=existing.return_count + max(0, int(source.return_count)),
        decayed_count=existing.decayed_count + max(0.0, float(source.decayed_count)),
        decayed_score_total=existing.decayed_score_total + float(source.decayed_score_total),
        score_total=existing.score_total + float(source.score_total),
        best_score=_max_optional_score(existing.best_score, source.best_score),
        completion_score_total=(
            existing.completion_score_total + float(source.completion_score_total)
        ),
        best_completion_score=_max_optional_score(
            existing.best_completion_score,
            source.best_completion_score,
        ),
        finish_score_total=existing.finish_score_total + float(source.finish_score_total),
        best_finish_score=_max_optional_score(existing.best_finish_score, source.best_finish_score),
        return_score_total=existing.return_score_total + float(source.return_score_total),
        best_return_score=_max_optional_score(existing.best_return_score, source.best_return_score),
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


def _objective_score(
    outcome: EngineTuningEpisodeOutcome,
    *,
    settings: BanditEngineTunerSettings,
) -> tuple[float | None, int | None]:
    finish_time_ms = successful_finish_time_ms(outcome)
    if settings.objective in {"finish_time", "safe_finish_time"}:
        return (
            None if finish_time_ms is None else finish_time_score(finish_time_ms),
            finish_time_ms,
        )
    return (1.0 if finish_time_ms is not None else 0.0, finish_time_ms)


def _selection_probabilities(
    *,
    projection: _EngineProjection,
    candidates: tuple[int, ...],
    uniform_exploration: float,
    safe_finish_rate_threshold: float,
    prior_finish_time_seconds: float,
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
    seed: int,
    draws: int,
) -> dict[int, float]:
    unobserved = tuple(
        engine_raw
        for engine_raw in candidates
        if _candidate_observation_count(
            projection.estimates[engine_raw],
            objective=projection.objective,
        )
        <= 0
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
        selected, _sampled_score = _sample_engine_setting(
            projection=projection,
            candidates=candidates,
            rng=rng,
            threshold=safe_finish_rate_threshold,
            prior_finish_time_seconds=prior_finish_time_seconds,
        )
        counts[selected] += 1
    return {engine_raw: counts[engine_raw] / draw_count for engine_raw in candidates}


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
        engine_raw
        for engine_raw in candidates
        if (
            sampled[engine_raw].finish_rate >= threshold
            and projection.estimates[engine_raw].finish_count > 0
        )
    )
    midpoint = _candidate_midpoint(candidates)
    if safe_candidates:
        selected = max(
            safe_candidates,
            key=lambda engine_raw: (
                sampled[engine_raw].finish_time_score,
                sampled[engine_raw].finish_rate,
                projection.estimates[engine_raw].finish_count,
                -abs(engine_raw - midpoint),
                -engine_raw,
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
    failures = max(0, estimate.episode_count - estimate.finish_count)
    finish_rate = rng.betavariate(1.0 + estimate.finish_count, 1.0 + failures)
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
    exploration_seconds: float,
) -> float:
    return max(0.25, float(exploration_seconds)) / max(1.0, candidate.decayed_count) ** 0.5


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


def _bandit_state_or_empty(
    state: EngineTuningRuntimeState | None,
    *,
    settings: BanditEngineTunerSettings,
) -> EngineTuningRuntimeState:
    if state is None:
        return empty_engine_tuning_state_for(
            objective=settings.objective,
            reward_fingerprint=settings.reward_fingerprint,
        )
    state = engine_tuning_state_with_objective(
        state,
        objective=settings.objective,
        reward_fingerprint=settings.reward_fingerprint,
        safe_finish_rate_threshold=settings.safe_finish_rate_threshold,
        prior_finish_time_seconds=settings.prior_finish_time_seconds,
    )
    return _canonical_bandit_state(state, settings=settings)
