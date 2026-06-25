# src/rl_fzerox/ui/watch/records/entry.py
"""Immutable per-track record entry update logic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.ui.watch.records.attempts import TrackAttemptStats
from rl_fzerox.ui.watch.records.finish import (
    _copy_setup,
    _finish_setup,
    _is_new_best_rank,
    _successful_finish_position,
    _successful_finish_time_ms,
)
from rl_fzerox.ui.watch.records.types import TrackFinishSetup


@dataclass(frozen=True, slots=True)
class TrackRecordEntry:
    best_finish_rank: int | None = None
    best_finish_rank_time_ms: int | None = None
    best_finish_rank_setup: TrackFinishSetup | None = None
    best_finish_time_ms: int | None = None
    best_finish_time_rank: int | None = None
    best_finish_time_setup: TrackFinishSetup | None = None
    latest_finish_rank: int | None = None
    latest_finish_time_ms: int | None = None
    latest_finish_delta_ms: int | None = None
    latest_finish_setup: TrackFinishSetup | None = None
    attempt_stats: TrackAttemptStats = field(default_factory=TrackAttemptStats)
    failed_attempt: bool = False

    def update(
        self,
        *,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        episode_done: bool,
    ) -> TrackRecordEntry:
        finish_time_ms = _successful_finish_time_ms(info, telemetry)
        finish_position = _successful_finish_position(info, telemetry)
        setup = _finish_setup(info, telemetry) if finish_time_ms is not None else None
        best_finish_time_ms = self.best_finish_time_ms
        best_finish_time_rank = self.best_finish_time_rank
        best_finish_time_setup = self.best_finish_time_setup
        latest_finish_rank = self.latest_finish_rank
        latest_finish_time_ms = self.latest_finish_time_ms
        latest_finish_delta_ms = self.latest_finish_delta_ms
        latest_finish_setup = self.latest_finish_setup
        if finish_time_ms is not None:
            latest_finish_rank = finish_position
            latest_finish_time_ms = finish_time_ms
            latest_finish_delta_ms = (
                None if best_finish_time_ms is None else finish_time_ms - best_finish_time_ms
            )
            latest_finish_setup = setup or latest_finish_setup
            if best_finish_time_ms is None or finish_time_ms < best_finish_time_ms:
                best_finish_time_ms = finish_time_ms
                best_finish_time_rank = finish_position
                best_finish_time_setup = setup or best_finish_time_setup

        best_finish_rank = self.best_finish_rank
        best_finish_rank_time_ms = self.best_finish_rank_time_ms
        best_finish_rank_setup = self.best_finish_rank_setup
        if (
            finish_time_ms is not None
            and finish_position is not None
            and _is_new_best_rank(
                current_rank=best_finish_rank,
                current_time_ms=best_finish_rank_time_ms,
                finish_position=finish_position,
                finish_time_ms=finish_time_ms,
            )
        ):
            best_finish_rank = finish_position
            best_finish_rank_time_ms = finish_time_ms
            best_finish_rank_setup = setup or best_finish_rank_setup

        failed_attempt = self.failed_attempt
        if episode_done:
            failed_attempt = info.get("termination_reason") != "finished"
        return TrackRecordEntry(
            best_finish_rank=best_finish_rank,
            best_finish_rank_time_ms=best_finish_rank_time_ms,
            best_finish_rank_setup=_copy_setup(best_finish_rank_setup),
            best_finish_time_ms=best_finish_time_ms,
            best_finish_time_rank=best_finish_time_rank,
            best_finish_time_setup=_copy_setup(best_finish_time_setup),
            latest_finish_rank=latest_finish_rank,
            latest_finish_time_ms=latest_finish_time_ms,
            latest_finish_delta_ms=latest_finish_delta_ms,
            latest_finish_setup=_copy_setup(latest_finish_setup),
            attempt_stats=self.attempt_stats.update(
                info=info,
                telemetry=telemetry,
                episode_done=episode_done,
            ),
            failed_attempt=failed_attempt,
        )
