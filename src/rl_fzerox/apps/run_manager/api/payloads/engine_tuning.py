# src/rl_fzerox/apps/run_manager/api/payloads/engine_tuning.py
"""API payloads for adaptive engine tuning."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b

from rl_fzerox.core.engine_tuning import (
    BanditEngineTunerSettings,
    EngineTunerBackend,
    EngineTunerSettings,
    EngineTuningCandidateEstimate,
    EngineTuningCandidateState,
    EngineTuningContext,
    EngineTuningRuntimeState,
    MlpEnsembleEngineTunerSettings,
    OrderedEngineTuner,
)
from rl_fzerox.core.engine_tuning.types import engine_bucket_candidates

ENGINE_TUNING_DISTRIBUTION_DRAWS = 128


def engine_tuning_state_payload(
    state: EngineTuningRuntimeState,
    *,
    settings: EngineTunerSettings | None = None,
) -> dict[str, object]:
    """Return a JSON-safe adaptive engine-tuning state payload."""

    payload_state = _payload_state(state, settings)
    return {
        "version": payload_state.version,
        "update_count": payload_state.update_count,
        "model_backend": _model_backend(payload_state, settings),
        "candidates": [
            _engine_tuning_candidate_payload(candidate) for candidate in payload_state.candidates
        ],
        "contexts": (
            [] if settings is None else _engine_tuning_context_payloads(payload_state, settings)
        ),
    }


def _payload_state(
    state: EngineTuningRuntimeState,
    settings: EngineTunerSettings | None,
) -> EngineTuningRuntimeState:
    if settings is None:
        return state
    if isinstance(settings, BanditEngineTunerSettings):
        state = _bandit_payload_state(state, settings)
    return OrderedEngineTuner(settings=settings, state=state).state


def _bandit_payload_state(
    state: EngineTuningRuntimeState,
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
        buckets[key] = _bucketed_payload_candidate(
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


def _exact_bucket(raw_value: int, candidates: tuple[int, ...]) -> int | None:
    raw_bucket = int(raw_value)
    if raw_bucket in candidates:
        return raw_bucket
    return None


def _bucketed_payload_candidate(
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


def _model_backend(
    state: EngineTuningRuntimeState,
    settings: EngineTunerSettings | None,
) -> EngineTunerBackend | None:
    if settings is not None:
        return settings.backend
    if state.model_state is None:
        return None
    return state.model_state.backend


def _engine_tuning_candidate_payload(
    candidate: EngineTuningCandidateState,
) -> dict[str, object]:
    return {
        "context_key": candidate.context_key,
        "course_key": candidate.course_key,
        "vehicle_id": candidate.vehicle_id,
        "engine_setting_raw_value": candidate.engine_setting_raw_value,
        "finish_count": candidate.finish_count,
        "mean_score": candidate.mean_score,
        "raw_mean_score": candidate.raw_mean_score,
        "best_score": candidate.best_score,
        "best_finish_time_ms": candidate.best_time_ms,
    }


def _engine_tuning_context_payloads(
    state: EngineTuningRuntimeState,
    settings: EngineTunerSettings,
) -> list[dict[str, object]]:
    tuner = OrderedEngineTuner(settings=settings, state=state)
    if isinstance(settings, MlpEnsembleEngineTunerSettings):
        return [
            _engine_tuning_context_payload(
                state=state,
                settings=settings,
                tuner=tuner,
                context=EngineTuningContext(
                    course_key=context.course_key,
                    vehicle_id=context.vehicle_id,
                ),
                finish_count=context.finish_count,
                observed_candidate_count=0,
            )
            for context in (() if state.model_state is None else state.model_state.contexts)
        ]
    return [
        _engine_tuning_context_payload(
            state=state,
            settings=settings,
            tuner=tuner,
            context=context,
            finish_count=sum(candidate.finish_count for candidate in candidates),
            observed_candidate_count=sum(
                1 for candidate in candidates if candidate.finish_count > 0
            ),
        )
        for context, candidates in _observed_contexts(state)
    ]


def _engine_tuning_context_payload(
    *,
    state: EngineTuningRuntimeState,
    settings: EngineTunerSettings,
    tuner: OrderedEngineTuner,
    context: EngineTuningContext,
    finish_count: int,
    observed_candidate_count: int,
) -> dict[str, object]:
    candidate_map = state.candidate_map()
    warmup_successes, model_ready = _model_readiness(settings, finish_count)
    estimates = tuner.distribution(
        context,
        seed=_distribution_seed(state, context.key),
        draws=ENGINE_TUNING_DISTRIBUTION_DRAWS,
    )
    recommendation = tuner.recommendation(context)
    return {
        "context_key": context.key,
        "course_key": context.course_key,
        "vehicle_id": context.vehicle_id,
        "finish_count": finish_count,
        "observed_candidate_count": observed_candidate_count,
        "model_ready": model_ready,
        "warmup_successes": warmup_successes,
        "warmup_remaining": max(0, warmup_successes - finish_count),
        "recommended_engine_setting_raw_value": recommendation.engine_setting_raw_value,
        "candidates": [
            _engine_tuning_candidate_estimate_payload(
                probability,
                candidate_map.get((context.key, probability.engine_setting_raw_value)),
            )
            for probability in estimates
        ],
    }


def _model_readiness(
    settings: EngineTunerSettings,
    finish_count: int,
) -> tuple[int, bool]:
    if isinstance(settings, MlpEnsembleEngineTunerSettings):
        warmup_successes = max(1, int(settings.warmup_successes))
        return warmup_successes, finish_count >= warmup_successes
    return 0, True


def _engine_tuning_candidate_estimate_payload(
    probability: EngineTuningCandidateEstimate,
    candidate: EngineTuningCandidateState | None,
) -> dict[str, object]:
    return {
        "engine_setting_raw_value": probability.engine_setting_raw_value,
        "selection_probability": probability.probability,
        "mean_score": probability.mean_score,
        "uncertainty_score": probability.uncertainty_score,
        "estimated_finish_time_ms": probability.estimated_finish_time_ms,
        "best_finish_time_ms": probability.best_finish_time_ms,
        "finish_count": 0 if candidate is None else candidate.finish_count,
    }


def _observed_contexts(
    state: EngineTuningRuntimeState,
) -> Iterable[tuple[EngineTuningContext, tuple[EngineTuningCandidateState, ...]]]:
    grouped: dict[str, list[EngineTuningCandidateState]] = {}
    for candidate in state.candidates:
        if candidate.finish_count <= 0:
            continue
        grouped.setdefault(candidate.context_key, []).append(candidate)
    contexts = []
    for candidates in grouped.values():
        first = candidates[0]
        contexts.append(
            (
                EngineTuningContext(course_key=first.course_key, vehicle_id=first.vehicle_id),
                tuple(sorted(candidates, key=lambda candidate: candidate.engine_setting_raw_value)),
            )
        )
    return sorted(
        contexts,
        key=lambda item: (
            -sum(candidate.finish_count for candidate in item[1]),
            item[0].course_key,
            item[0].vehicle_id,
        ),
    )


def _distribution_seed(state: EngineTuningRuntimeState, context_key: str) -> int:
    data = f"{state.version}:{state.update_count}:{context_key}".encode()
    return int.from_bytes(blake2b(data, digest_size=8).digest(), "big")
