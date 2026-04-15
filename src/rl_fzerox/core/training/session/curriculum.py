# src/rl_fzerox/core/training/session/curriculum.py
from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
)


class ActionMaskCurriculumController:
    """Track episode-smoothed curriculum progress and stage promotion."""

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config
        self._race_laps_completed_window: deque[float] = deque(maxlen=config.smoothing_episodes)
        self._milestones_completed_window: deque[float] = deque(maxlen=config.smoothing_episodes)
        self._stage_episode_count = 0
        self._stage_index = 0 if config.enabled and config.stages else None

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
            self._race_laps_completed_window.append(_episode_metric(episode, "race_laps_completed"))
            self._milestones_completed_window.append(
                _episode_metric(episode, "milestones_completed")
            )
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
            return mean_value is not None and mean_value >= float(
                trigger.race_laps_completed_mean_gte
            )
        if trigger.milestones_completed_mean_gte is not None:
            mean_value = _window_mean(self._milestones_completed_window)
            return mean_value is not None and mean_value >= float(
                trigger.milestones_completed_mean_gte
            )
        return False

    def _clear_stage_windows(self) -> None:
        self._race_laps_completed_window.clear()
        self._milestones_completed_window.clear()
        self._stage_episode_count = 0


def _episode_metric(episode: dict[str, object], key: str) -> float:
    value = episode.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _window_mean(window: Sequence[float]) -> float | None:
    if len(window) == 0:
        return None
    return sum(window) / len(window)
