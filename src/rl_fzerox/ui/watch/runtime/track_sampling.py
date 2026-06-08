# src/rl_fzerox/ui/watch/runtime/track_sampling.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_track_sampling_from_state,
)
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.core.training.runs import continue_run_paths, materialize_train_run_config
from rl_fzerox.core.training.session.callbacks.track_sampling import TrackSamplingRuntimeState

TrackSamplingSignature = tuple[tuple[object, ...], ...]
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedTrackSamplingRefresh:
    """Refresh mutable manager-owned X Cup slots for a running watch worker."""

    store: ManagerStore
    run_id: str
    interval_seconds: float = 10.0
    _last_check_monotonic: float = field(default=0.0, init=False)

    @classmethod
    def from_config(cls, config: WatchAppConfig) -> ManagedTrackSamplingRefresh | None:
        db_path = config.watch.manager_db_path
        run_id = config.watch.managed_run_id
        if db_path is None or not run_id:
            return None
        return cls(
            store=ManagerStore(Path(db_path).expanduser().resolve()),
            run_id=run_id,
        )

    def refreshed_config(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingConfig | None:
        if not force and not self._refresh_due():
            return None
        self._last_check_monotonic = time.monotonic()
        state = self.store.get_run_track_sampling_state(self.run_id)
        projected_config = restore_generated_x_cup_track_sampling_from_state(
            current_config,
            state=state,
        )
        projected_signature = _generated_x_cup_signature(projected_config)
        if projected_signature == _generated_x_cup_signature(current_config):
            return None
        refreshed_config = self._materialized_track_sampling(state=state)
        if refreshed_config is None or not _track_sampling_baselines_ready(refreshed_config):
            LOGGER.warning(
                "skipping managed watch track-sampling refresh for run_id=%s: "
                "generated X Cup state changed but materialized baselines are not ready",
                self.run_id,
            )
            return None
        if _generated_x_cup_signature(refreshed_config) != projected_signature:
            LOGGER.warning(
                "skipping managed watch track-sampling refresh for run_id=%s: "
                "materialized config does not match current runtime generation",
                self.run_id,
            )
            return None
        return refreshed_config

    def _refresh_due(self) -> bool:
        return time.monotonic() - self._last_check_monotonic >= self.interval_seconds

    def _materialized_track_sampling(
        self,
        *,
        state: TrackSamplingRuntimeState | None,
    ) -> TrackSamplingConfig | None:
        run = self.store.get_run(self.run_id)
        if run is None:
            return None
        train_config = build_managed_train_app_config(
            run.config,
            run_id=run.id,
            run_dir=run.run_dir,
        )
        train_config = train_config.model_copy(
            update={
                "env": train_config.env.model_copy(
                    update={
                        "track_sampling": restore_generated_x_cup_track_sampling_from_state(
                            train_config.env.track_sampling,
                            state=state,
                        )
                    }
                )
            }
        )
        try:
            materialized = materialize_train_run_config(
                train_config,
                run_paths=continue_run_paths(run.run_dir),
            )
        except Exception:
            LOGGER.exception(
                "managed watch track-sampling refresh failed for run_id=%s",
                self.run_id,
            )
            return None
        return materialized.env.track_sampling


def _generated_x_cup_signature(config: TrackSamplingConfig) -> TrackSamplingSignature:
    return tuple(
        (
            entry.generated_course_slot,
            entry.runtime_course_key,
            entry.id,
            entry.course_id,
            entry.generated_course_hash,
            entry.generated_course_seed,
            entry.generated_course_generation,
        )
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
    )


def _track_sampling_baselines_ready(config: TrackSamplingConfig) -> bool:
    if not config.enabled:
        return True
    return all(
        entry.baseline_state_path is not None
        and entry.baseline_state_path.expanduser().resolve().is_file()
        for entry in config.entries
    )
