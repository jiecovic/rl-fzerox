# src/rl_fzerox/apps/run_manager/api/payloads/engine_tuning.py
"""API payloads for adaptive engine tuning."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b

from rl_fzerox.core.engine_tuning import (
    AdaptiveEngineBandit,
    EngineBanditSettings,
    EngineTuningArmState,
    EngineTuningBinProbability,
    EngineTuningContext,
    EngineTuningRuntimeState,
)

ENGINE_TUNING_DISTRIBUTION_DRAWS = 512


def engine_tuning_state_payload(
    state: EngineTuningRuntimeState,
    *,
    settings: EngineBanditSettings | None = None,
) -> dict[str, object]:
    """Return a JSON-safe adaptive engine-tuning state payload."""

    return {
        "version": state.version,
        "update_count": state.update_count,
        "arms": [_engine_tuning_arm_payload(arm) for arm in state.arms],
        "contexts": [] if settings is None else _engine_tuning_context_payloads(state, settings),
    }


def _engine_tuning_arm_payload(arm: EngineTuningArmState) -> dict[str, object]:
    return {
        "context_key": arm.context_key,
        "course_key": arm.course_key,
        "vehicle_id": arm.vehicle_id,
        "engine_setting_raw_value": arm.engine_setting_raw_value,
        "attempts": arm.attempts,
        "finished_attempts": arm.finished_attempts,
        "finish_rate": arm.finish_rate,
        "mean_completion": arm.mean_completion,
        "mean_score": arm.mean_score,
        "raw_mean_score": arm.raw_mean_score,
        "best_score": arm.best_score,
    }


def _engine_tuning_context_payloads(
    state: EngineTuningRuntimeState,
    settings: EngineBanditSettings,
) -> list[dict[str, object]]:
    bandit = AdaptiveEngineBandit(settings=settings, state=state)
    return [
        _engine_tuning_context_payload(
            state=state,
            bandit=bandit,
            context=context,
            arms=arms,
        )
        for context, arms in _observed_contexts(state)
    ]


def _engine_tuning_context_payload(
    *,
    state: EngineTuningRuntimeState,
    bandit: AdaptiveEngineBandit,
    context: EngineTuningContext,
    arms: tuple[EngineTuningArmState, ...],
) -> dict[str, object]:
    arm_map = state.arm_map()
    recommendation = bandit.recommendation(context)
    return {
        "context_key": context.key,
        "course_key": context.course_key,
        "vehicle_id": context.vehicle_id,
        "attempts": sum(arm.attempts for arm in arms),
        "observed_arm_count": sum(1 for arm in arms if arm.attempts > 0),
        "recommended_engine_setting_raw_value": recommendation.engine_setting_raw_value,
        "bins": [
            _engine_tuning_bin_payload(
                probability,
                arm_map.get((context.key, probability.engine_setting_raw_value)),
            )
            for probability in bandit.choice_distribution(
                context,
                seed=_distribution_seed(state, context.key),
                draws=ENGINE_TUNING_DISTRIBUTION_DRAWS,
            )
        ],
    }


def _engine_tuning_bin_payload(
    probability: EngineTuningBinProbability,
    arm: EngineTuningArmState | None,
) -> dict[str, object]:
    return {
        "engine_setting_raw_value": probability.engine_setting_raw_value,
        "selection_probability": probability.probability,
        "posterior_mean": probability.posterior_mean,
        "attempts": probability.attempts,
        "finish_rate": None if arm is None else arm.finish_rate,
        "mean_completion": None if arm is None else arm.mean_completion,
    }


def _observed_contexts(
    state: EngineTuningRuntimeState,
) -> Iterable[tuple[EngineTuningContext, tuple[EngineTuningArmState, ...]]]:
    grouped: dict[str, list[EngineTuningArmState]] = {}
    for arm in state.arms:
        if arm.attempts <= 0:
            continue
        grouped.setdefault(arm.context_key, []).append(arm)
    contexts = []
    for arms in grouped.values():
        first = arms[0]
        contexts.append(
            (
                EngineTuningContext(course_key=first.course_key, vehicle_id=first.vehicle_id),
                tuple(sorted(arms, key=lambda arm: arm.engine_setting_raw_value)),
            )
        )
    return sorted(
        contexts,
        key=lambda item: (
            -sum(arm.attempts for arm in item[1]),
            item[0].course_key,
            item[0].vehicle_id,
        ),
    )


def _distribution_seed(state: EngineTuningRuntimeState, context_key: str) -> int:
    data = f"{state.version}:{state.update_count}:{context_key}".encode()
    return int.from_bytes(blake2b(data, digest_size=8).digest(), "big")
