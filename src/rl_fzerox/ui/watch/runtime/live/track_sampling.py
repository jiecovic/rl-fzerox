# src/rl_fzerox/ui/watch/runtime/live/track_sampling.py
"""Live-worker track-sampling state and refresh side effects.

The live worker owns orchestration, but track-sampling refresh has its own
mutable state: active config, course rotation, selected reset target, and
locked reset target. Keeping that state here avoids spreading refresh side
effects through the main worker loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.ui.watch.runtime.courses.navigation import WatchCourseRotation
from rl_fzerox.ui.watch.runtime.courses.sampling import (
    ManagedTrackSamplingRefresh,
    TrackSamplingRefreshStatus,
    unmaterialized_generated_x_cup_entries,
)

LOGGER = logging.getLogger(__name__)


class _TrackSamplingEnv(Protocol):
    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None: ...

    def set_locked_reset_course(self, target_key: str | None) -> None: ...

    def preload_track_baseline_paths(self, paths: tuple[Path, ...]) -> None: ...


class _TrackSamplingRefresh(Protocol):
    def refreshed_config(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingConfig | None: ...

    def refresh_status(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingRefreshStatus: ...


@dataclass(slots=True)
class LiveTrackSamplingState:
    active_track_sampling: TrackSamplingConfig
    course_rotation: WatchCourseRotation
    selected_reset_target_key: str | None
    locked_reset_target_key: str | None
    refresh_source: _TrackSamplingRefresh | None

    @classmethod
    def from_config(cls, config: WatchAppConfig) -> LiveTrackSamplingState:
        active_track_sampling = config.env.track_sampling
        course_rotation = WatchCourseRotation.from_entries(active_track_sampling.entries)
        return cls(
            active_track_sampling=active_track_sampling,
            course_rotation=course_rotation,
            selected_reset_target_key=course_rotation.normalized_key(None),
            locked_reset_target_key=None,
            refresh_source=ManagedTrackSamplingRefresh.from_config(config),
        )

    def refresh(self, env: _TrackSamplingEnv, *, force: bool = False) -> bool:
        if self.refresh_source is None:
            return False
        refreshed_track_sampling = self.refresh_source.refreshed_config(
            self.active_track_sampling,
            force=force,
        )
        if refreshed_track_sampling is None:
            return False
        return self._apply_refreshed_config(env, refreshed_track_sampling)

    def ready_for_reset(self, env: _TrackSamplingEnv, *, force: bool = False) -> bool:
        if self.refresh_source is None:
            return True
        status = self.refresh_source.refresh_status(self.active_track_sampling, force=force)
        if status.refreshed_config is not None:
            self._apply_refreshed_config(env, status.refreshed_config)
        return (
            status.ready_for_reset
            and _generated_x_cup_has_materialized_entries(self.active_track_sampling)
        )

    def preload_generated_x_cup_baselines(self, env: _TrackSamplingEnv) -> None:
        """Warm active generated-course baselines before interactive navigation."""

        paths = _generated_x_cup_baseline_paths(self.active_track_sampling)
        if paths:
            env.preload_track_baseline_paths(paths)

    def select_current_target(self, info: dict[str, object]) -> None:
        current_target = self.course_rotation.target_for_info(info)
        if current_target is not None:
            self.selected_reset_target_key = current_target.key

    def _apply_refreshed_config(
        self,
        env: _TrackSamplingEnv,
        refreshed_track_sampling: TrackSamplingConfig,
    ) -> bool:
        if not _generated_x_cup_has_materialized_entries(refreshed_track_sampling):
            LOGGER.warning(
                "skipping watch track-sampling refresh because generated X Cup "
                "baselines are not runtime-ready"
            )
            return False
        try:
            paths = _generated_x_cup_baseline_paths(refreshed_track_sampling)
            if paths:
                env.preload_track_baseline_paths(paths)
        except OSError as exc:
            LOGGER.warning(
                "skipping watch track-sampling refresh because generated X Cup "
                "baseline preload failed: %s",
                exc,
            )
            return False
        env.set_track_sampling_config(refreshed_track_sampling)
        self.active_track_sampling = refreshed_track_sampling
        self.course_rotation = WatchCourseRotation.from_entries(self.active_track_sampling.entries)
        self.selected_reset_target_key = self.course_rotation.normalized_key(
            self.selected_reset_target_key
        )
        if self.course_rotation.target_by_key(self.locked_reset_target_key) is None:
            self.locked_reset_target_key = None
            env.set_locked_reset_course(None)
        return True


def _generated_x_cup_has_materialized_entries(config: TrackSamplingConfig) -> bool:
    # Disk existence is checked before a refresh is applied. Once preloaded,
    # active X Cup entries stay usable from the reset cache even if files move.
    return not unmaterialized_generated_x_cup_entries(config)


def _generated_x_cup_baseline_paths(config: TrackSamplingConfig) -> tuple[Path, ...]:
    return tuple(
        path.expanduser().resolve()
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and (path := entry.baseline_state_path) is not None
    )
