# src/rl_fzerox/core/engine_tuning/training.py
"""Training-time adaptive engine-tuning controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from zlib import crc32

from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
from rl_fzerox.core.engine_tuning.sampling import (
    EngineTuningResetCandidate,
    EngineTuningResetContext,
    EngineTuningResetSampler,
)
from rl_fzerox.core.engine_tuning.state import (
    EngineTuningRuntimeState,
    empty_engine_tuning_state,
)
from rl_fzerox.core.engine_tuning.tuner import (
    EngineTuningContext,
    EngineTuningEpisodeOutcome,
    OrderedEngineTuner,
)
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig

RESET_SAMPLER_DISTRIBUTION_DRAWS = 64


@dataclass(slots=True)
class EngineTuningTrainingController:
    """Own the mutable adaptive engine tuner used by the trainer callback."""

    config: AdaptiveEngineTuningConfig
    state: EngineTuningRuntimeState | None = None
    _tuner: OrderedEngineTuner = field(init=False, repr=False)
    _rollout_outcomes: list[EngineTuningEpisodeOutcome] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._tuner = OrderedEngineTuner(
            settings=engine_tuner_settings(self.config),
            state=self.state or empty_engine_tuning_state(),
        )

    @property
    def runtime_state(self) -> EngineTuningRuntimeState:
        return self._tuner.state

    def reset_sampler_snapshot(
        self,
        contexts: Sequence[EngineTuningContext],
    ) -> EngineTuningResetSampler:
        """Build the plain engine-choice table sent to reset workers."""

        reset_contexts: list[EngineTuningResetContext] = []
        for context in _unique_contexts(contexts):
            estimates = self._tuner.distribution(
                context,
                seed=_distribution_seed(self.runtime_state.update_count, context),
                draws=RESET_SAMPLER_DISTRIBUTION_DRAWS,
            )
            if not estimates:
                continue
            recommendation = self._tuner.recommendation(context)
            reset_contexts.append(
                EngineTuningResetContext(
                    context=context,
                    candidates=tuple(
                        EngineTuningResetCandidate(
                            engine_setting_raw_value=estimate.engine_setting_raw_value,
                            probability=estimate.probability,
                            mean_score=estimate.mean_score,
                            sampled_score=estimate.uncertainty_score,
                            finish_count=estimate.finish_count,
                            estimated_finish_time_ms=estimate.estimated_finish_time_ms,
                            best_finish_time_ms=estimate.best_finish_time_ms,
                        )
                        for estimate in estimates
                    ),
                    greedy_engine_setting_raw_value=(
                        recommendation.engine_setting_raw_value
                    ),
                )
            )
        return EngineTuningResetSampler(contexts=tuple(reset_contexts))

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> bool:
        """Record terminal episode dictionaries and return whether state changed."""

        outcomes: list[EngineTuningEpisodeOutcome] = []
        for episode in episodes:
            outcome = engine_tuning_outcome_from_episode(episode)
            if outcome is None:
                continue
            outcomes.append(outcome)
        if not outcomes:
            return False
        if self.config.backend == "mlp_ensemble":
            self._rollout_outcomes.extend(outcomes)
            return False
        previous_state = self._tuner.state
        return self._tuner.record_many(tuple(outcomes)) is not previous_state

    def record_rollout_episodes(self) -> bool:
        """Train rollout-scoped tuner backends and return whether state changed."""

        if self.config.backend != "mlp_ensemble" or not self._rollout_outcomes:
            return False
        outcomes = tuple(self._rollout_outcomes)
        self._rollout_outcomes.clear()
        previous_state = self._tuner.state
        return self._tuner.record_many(outcomes) is not previous_state

    def log_values(self) -> dict[str, float]:
        """Return compact TensorBoard metrics for successful engine observations."""

        values: dict[str, float] = {}
        for candidate in self.runtime_state.candidates:
            key = _sanitize_log_key(candidate.context_key)
            suffix = f"{key}/engine_{candidate.engine_setting_raw_value}"
            if candidate.mean_score is not None:
                values[f"engine_tuning/{suffix}/mean_score"] = candidate.mean_score
            if candidate.best_time_ms is not None:
                values[f"engine_tuning/{suffix}/best_time_ms"] = float(candidate.best_time_ms)
            values[f"engine_tuning/{suffix}/finishes"] = float(candidate.finish_count)
        values["engine_tuning/update_count"] = float(self.runtime_state.update_count)
        return values


def engine_tuning_outcome_from_episode(
    episode: Mapping[str, object],
) -> EngineTuningEpisodeOutcome | None:
    """Build one engine-tuning outcome from an SB3 Monitor episode dictionary."""

    if _uses_alt_baseline_sample(episode):
        return None
    context_key = _mapping_str(episode, "engine_tuning_context_key")
    course_key = _mapping_str(episode, "engine_tuning_course_key")
    vehicle_id = _mapping_str(episode, "engine_tuning_vehicle_id")
    engine_raw = _mapping_int(episode, "track_engine_setting_raw_value")
    if context_key is None or course_key is None or vehicle_id is None or engine_raw is None:
        return None
    context = EngineTuningContext(
        course_key=course_key,
        vehicle_id=vehicle_id,
    )
    if context.key != context_key:
        return None
    return EngineTuningEpisodeOutcome(
        context=context,
        engine_setting_raw_value=engine_raw,
        completion_fraction=_episode_completion_fraction(episode),
        finished=episode.get("termination_reason") == "finished",
        race_time_ms=_mapping_optional_int(episode, "race_time_ms"),
        finish_position=_mapping_optional_int(episode, "position"),
        total_racers=_mapping_optional_int(episode, "total_racers"),
    )


def _episode_completion_fraction(episode: Mapping[str, object]) -> float:
    value = _mapping_float(episode, "episode_completion_fraction")
    if value is not None:
        return max(0.0, min(1.0, value))
    return 1.0 if episode.get("termination_reason") == "finished" else 0.0


def _uses_alt_baseline_sample(episode: Mapping[str, object]) -> bool:
    value = episode.get("track_alt_baseline_id")
    return isinstance(value, str) and bool(value.strip())


def _mapping_str(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _mapping_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _mapping_optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    return _mapping_int(raw, key)


def _mapping_float(raw: Mapping[str, object], key: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _sanitize_log_key(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return sanitized.strip("_") or "unknown"


def _unique_contexts(
    contexts: Sequence[EngineTuningContext],
) -> tuple[EngineTuningContext, ...]:
    by_key: dict[str, EngineTuningContext] = {}
    for context in contexts:
        by_key.setdefault(context.key, context)
    return tuple(by_key[key] for key in sorted(by_key))


def _distribution_seed(update_count: int, context: EngineTuningContext) -> int:
    data = f"{int(update_count)}|{context.key}".encode()
    return crc32(data) & 0xFFFF_FFFF
