# src/rl_fzerox/ui/watch/runtime/track_sampling.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.projection.x_cup_runtime import (
    restore_generated_x_cup_track_sampling_from_state,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig

TrackSamplingSignature = tuple[tuple[object, ...], ...]


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
        refreshed_config = restore_generated_x_cup_track_sampling_from_state(
            current_config,
            state=state,
        )
        if _generated_x_cup_signature(refreshed_config) == _generated_x_cup_signature(
            current_config
        ):
            return None
        return refreshed_config

    def _refresh_due(self) -> bool:
        return time.monotonic() - self._last_check_monotonic >= self.interval_seconds


def _generated_x_cup_signature(config: TrackSamplingConfig) -> TrackSamplingSignature:
    return tuple(
        (
            entry.generated_course_slot,
            entry.runtime_course_key,
            entry.id,
            entry.course_id,
            None if entry.baseline_state_path is None else str(entry.baseline_state_path),
            entry.generated_course_hash,
            entry.generated_course_seed,
            entry.generated_course_generation,
        )
        for entry in config.entries
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind
    )
