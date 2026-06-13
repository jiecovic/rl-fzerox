# src/rl_fzerox/ui/watch/runtime/track_sampling.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_track_sampling_artifacts,
    restore_generated_x_cup_track_sampling_from_slots,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    apply_alt_baselines_to_track_sampling,
)

TrackSamplingSignature = tuple[tuple[object, ...], ...]
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackSamplingRefreshStatus:
    """Result of checking manager-owned track-sampling state."""

    refreshed_config: TrackSamplingConfig | None
    ready_for_reset: bool


@dataclass(slots=True)
class ManagedTrackSamplingRefresh:
    """Refresh mutable manager-owned X Cup slots for a running watch worker."""

    store: ManagerStore
    run_id: str
    interval_seconds: float = 10.0
    _last_check_monotonic: float = field(default=0.0, init=False)
    _last_blocked_signature: TrackSamplingSignature | None = field(default=None, init=False)

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
        return self.refresh_status(current_config, force=force).refreshed_config

    def refresh_status(
        self,
        current_config: TrackSamplingConfig,
        *,
        force: bool = False,
    ) -> TrackSamplingRefreshStatus:
        if not force and not self._refresh_due():
            return TrackSamplingRefreshStatus(
                refreshed_config=None,
                ready_for_reset=self._last_blocked_signature is None,
            )
        self._last_check_monotonic = time.monotonic()
        slots = self.store.get_run_generated_x_cup_slots(self.run_id)
        projected_config = restore_generated_x_cup_track_sampling_from_slots(
            current_config,
            slots=slots,
        )
        projected_signature = _generated_x_cup_signature(projected_config)
        has_materialized_x_cup_entries = not unmaterialized_generated_x_cup_entries(current_config)
        refreshed_config = projected_config
        if (
            projected_signature == _generated_x_cup_signature(current_config)
            and has_materialized_x_cup_entries
            and not missing_generated_x_cup_baseline_paths(current_config)
        ):
            self._last_blocked_signature = None
        else:
            materialized_config = self._with_materialized_x_cup_artifacts(projected_config)
            if materialized_config is None:
                if projected_signature != self._last_blocked_signature:
                    LOGGER.warning(
                        "skipping managed watch track-sampling refresh for run_id=%s: "
                        "generated X Cup state changed but run-local baseline artifacts "
                        "are not ready",
                        self.run_id,
                    )
                    self._last_blocked_signature = projected_signature
                return TrackSamplingRefreshStatus(refreshed_config=None, ready_for_reset=False)
            refreshed_config = materialized_config
            self._last_blocked_signature = None
        refreshed_config = apply_alt_baselines_to_track_sampling(
            refreshed_config,
            self.store.active_run_alt_baselines(self.run_id),
        )
        if _track_sampling_signature(refreshed_config) == _track_sampling_signature(current_config):
            return TrackSamplingRefreshStatus(refreshed_config=None, ready_for_reset=True)
        self._last_blocked_signature = None
        return TrackSamplingRefreshStatus(refreshed_config=refreshed_config, ready_for_reset=True)

    def _refresh_due(self) -> bool:
        return time.monotonic() - self._last_check_monotonic >= self.interval_seconds

    def _with_materialized_x_cup_artifacts(
        self,
        config: TrackSamplingConfig,
    ) -> TrackSamplingConfig | None:
        refreshed = restore_generated_x_cup_track_sampling_artifacts(
            config,
            artifacts=self.store.get_run_track_sampling_artifacts(self.run_id),
        )
        if unmaterialized_generated_x_cup_entries(refreshed):
            return None
        missing_paths = missing_generated_x_cup_baseline_paths(refreshed)
        return None if missing_paths else refreshed


def missing_generated_x_cup_baseline_paths(config: TrackSamplingConfig) -> tuple[Path, ...]:
    """Return generated X Cup state paths that disappeared from disk."""

    return tuple(
        path
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and (path := _resolved_baseline_state_path(entry)) is not None
        and not path.is_file()
    )


def unmaterialized_generated_x_cup_entries(
    config: TrackSamplingConfig,
) -> tuple[TrackSamplingEntryConfig, ...]:
    """Return generated X Cup entries that do not have a DB-owned state path."""

    return tuple(
        entry
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
        and entry.baseline_state_path is None
    )


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


def _track_sampling_signature(config: TrackSamplingConfig) -> TrackSamplingSignature:
    return tuple(
        (
            entry.id,
            entry.baseline_state_path,
            float(entry.weight),
            entry.baseline_group_id,
            entry.baseline_group_weight,
            entry.alt_baseline_id,
            entry.alt_baseline_label,
            entry.alt_baseline_source_entry_id,
            entry.generated_course_slot,
            entry.generated_course_generation,
            entry.generated_course_hash,
        )
        for entry in config.entries
    )


def _resolved_baseline_state_path(entry: TrackSamplingEntryConfig) -> Path | None:
    path = entry.baseline_state_path
    return None if path is None else path.expanduser().resolve()
