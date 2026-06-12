# src/rl_fzerox/core/engine_tuning/training.py
"""Training-time adaptive engine-tuning controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
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


@dataclass(slots=True)
class EngineTuningTrainingController:
    """Own the mutable adaptive engine tuner used by the trainer callback."""

    config: AdaptiveEngineTuningConfig
    state: EngineTuningRuntimeState | None = None
    _tuner: OrderedEngineTuner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tuner = OrderedEngineTuner(
            settings=engine_tuner_settings(self.config),
            state=self.state or empty_engine_tuning_state(),
        )

    @property
    def runtime_state(self) -> EngineTuningRuntimeState:
        return self._tuner.state

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> bool:
        """Record terminal episode dictionaries and return whether state changed."""

        changed = False
        for episode in episodes:
            outcome = engine_tuning_outcome_from_episode(episode)
            if outcome is None:
                continue
            previous_state = self._tuner.state
            if self._tuner.record(outcome) is not previous_state:
                changed = True
        return changed

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
