# src/rl_fzerox/core/training/session/curriculum.py
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
    EnvConfig,
    PerTrackLapsCompletedTriggerConfig,
    TrackSamplingConfig,
)


class ActionMaskCurriculumController:
    """Track episode-smoothed curriculum progress and stage promotion."""

    def __init__(
        self,
        config: CurriculumConfig,
        *,
        env_config: EnvConfig | None = None,
        initial_stage_index: int | None = None,
    ) -> None:
        self._config = config
        self._base_track_ids = _track_ids_from_sampling(
            None if env_config is None else env_config.track_sampling
        )
        self._race_laps_completed_window: deque[float] = deque(maxlen=config.smoothing_episodes)
        self._race_laps_completed_by_track: dict[str, deque[float]] = {}
        self._stage_episode_count = 0
        self._stage_index = 0 if config.enabled and config.stages else None
        if initial_stage_index is not None:
            self._set_initial_stage(initial_stage_index)

    @property
    def enabled(self) -> bool:
        """Return whether this controller has any active curriculum stages."""

        return self._stage_index is not None

    @property
    def stage_index(self) -> int | None:
        """Return the currently active curriculum stage index."""

        return self._stage_index

    @property
    def stage_name(self) -> str | None:
        """Return the current curriculum stage name, if any."""

        if self._stage_index is None:
            return None
        return self._config.stages[self._stage_index].name

    @property
    def stage_train_overrides(self) -> CurriculumTrainOverridesConfig | None:
        """Return train overrides for the current stage, if configured."""

        if self._stage_index is None:
            return None
        return self._config.stages[self._stage_index].train

    def record_episodes(self, episodes: Sequence[dict[str, object]]) -> int | None:
        """Update the smoothing windows and return a promoted stage index, if any."""

        if self._stage_index is None or not episodes:
            return None

        for episode in episodes:
            race_laps_completed = _episode_metric(episode, "race_laps_completed")
            self._race_laps_completed_window.append(race_laps_completed)
            track_id = _episode_track_id(episode)
            if track_id is not None:
                self._race_laps_completed_by_track.setdefault(
                    track_id,
                    deque(maxlen=self._config.smoothing_episodes),
                ).append(race_laps_completed)
            self._stage_episode_count += 1

        stage = self._config.stages[self._stage_index]
        if not self._can_promote(stage):
            return None
        if self._stage_index >= len(self._config.stages) - 1:
            return None

        self._stage_index += 1
        self._clear_stage_windows()
        return self._stage_index

    def _can_promote(self, stage: CurriculumStageConfig) -> bool:
        trigger = stage.until
        if trigger is None:
            return False
        if self._stage_episode_count < self._config.min_stage_episodes:
            return False

        if trigger.race_laps_completed_mean_gte is not None:
            mean_value = _window_mean(self._race_laps_completed_window)
            if mean_value is None or mean_value < float(trigger.race_laps_completed_mean_gte):
                return False
        if trigger.per_track_laps_completed is not None and not self._per_track_trigger_passes(
            stage,
            trigger.per_track_laps_completed,
        ):
            return False
        return True

    def _per_track_trigger_passes(
        self,
        stage: CurriculumStageConfig,
        trigger: PerTrackLapsCompletedTriggerConfig,
    ) -> bool:
        active_track_ids = self._active_track_ids(stage)
        if not active_track_ids:
            return False

        passing_tracks = 0
        for track_id in active_track_ids:
            window = self._race_laps_completed_by_track.get(track_id)
            if window is None or len(window) < trigger.min_episodes_per_track:
                continue
            mean_value = _window_mean(window)
            if mean_value is not None and mean_value >= trigger.mean_gte:
                passing_tracks += 1

        return passing_tracks / len(active_track_ids) >= trigger.min_track_fraction_gte

    def _active_track_ids(self, stage: CurriculumStageConfig) -> tuple[str, ...]:
        stage_track_ids = _track_ids_from_sampling(stage.track_sampling)
        return stage_track_ids or self._base_track_ids

    def _clear_stage_windows(self) -> None:
        self._race_laps_completed_window.clear()
        self._race_laps_completed_by_track.clear()
        self._stage_episode_count = 0

    def _set_initial_stage(self, stage_index: int) -> None:
        if self._stage_index is None:
            raise ValueError("Cannot restore a curriculum stage without configured stages")
        if not 0 <= stage_index < len(self._config.stages):
            raise ValueError(f"Invalid initial curriculum stage index {stage_index}")
        self._stage_index = int(stage_index)


def _episode_metric(episode: dict[str, object], key: str) -> float:
    value = episode.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _episode_track_id(episode: dict[str, object]) -> str | None:
    for key in ("track_course_id", "track_id", "track_course_name", "course_index"):
        value = episode.get(key)
        if isinstance(value, int):
            return f"course_{value}"
        if isinstance(value, str) and value.strip():
            return value
    return None


def _track_ids_from_sampling(track_sampling: TrackSamplingConfig | None) -> tuple[str, ...]:
    if track_sampling is None:
        return ()

    track_ids: list[str] = []
    seen: set[str] = set()
    for entry in track_sampling.entries:
        track_id = entry.course_id or entry.id
        if track_id is None or track_id in seen:
            continue
        track_ids.append(track_id)
        seen.add(track_id)
    return tuple(track_ids)


def _window_mean(window: Sequence[float]) -> float | None:
    if len(window) == 0:
        return None
    return sum(window) / len(window)
