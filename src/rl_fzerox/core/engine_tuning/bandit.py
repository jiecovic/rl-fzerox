# src/rl_fzerox/core/engine_tuning/bandit.py
"""Aggregate bandit backend for adaptive engine tuning.

The bandit is the maintained reset-time tuner. It aggregates default-baseline
episodes per course/vehicle/engine bucket and can optimize either raw finish
time, finish rate, or a safe-finish-time objective that gates speed by
reliability.
"""

from __future__ import annotations

from random import Random

from rl_fzerox.core.engine_tuning.bandit_sampling import (
    _bandit_greedy_engine_setting,
    _candidate_estimate,
    _candidate_observation_count,
    _candidate_uncertainty,
    _EngineEstimate,
    _EngineProjection,
    _sample_engine_setting,
    _selection_probabilities,
)
from rl_fzerox.core.engine_tuning.state import (
    EngineTuningCandidateState,
    EngineTuningRuntimeState,
    empty_engine_tuning_state_for,
    engine_tuning_state_with_objective,
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
    """Choose from coarse engine buckets using observed episode aggregates."""

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
                uncertainty_score=_prior_uncertainty(
                    objective=self._settings.objective,
                    exploration_seconds=self._settings.exploration_seconds,
                ),
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
                uncertainty_score=_prior_uncertainty(
                    objective=self._settings.objective,
                    exploration_seconds=self._settings.exploration_seconds,
                ),
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
                objective=self._settings.objective,
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


def _prior_uncertainty(
    *,
    objective: str,
    exploration_seconds: float,
) -> float:
    if objective == "finish_rate":
        return (1.0 / 12.0) ** 0.5
    return max(0.0, float(exploration_seconds))


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
