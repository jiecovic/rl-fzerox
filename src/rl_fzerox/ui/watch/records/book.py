# src/rl_fzerox/ui/watch/records/book.py
"""Immutable record-book container used by Watch runtime snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.ui.watch.records.entry import TrackRecordEntry
from rl_fzerox.ui.watch.records.finish import _copy_setup, _updated_best_finish_position
from rl_fzerox.ui.watch.records.identity import track_record_key, track_record_lookup_keys
from rl_fzerox.ui.watch.records.types import TrackRecordKey


@dataclass(frozen=True, slots=True)
class TrackRecordBook:
    entries: dict[TrackRecordKey, TrackRecordEntry] = field(default_factory=dict)
    best_finish_position: int | None = None

    @property
    def is_empty(self) -> bool:
        return self.best_finish_position is None and not self.entries

    def snapshot(self) -> TrackRecordBook:
        return TrackRecordBook(
            entries={
                key: TrackRecordEntry(
                    best_finish_rank=entry.best_finish_rank,
                    best_finish_rank_time_ms=entry.best_finish_rank_time_ms,
                    best_finish_rank_setup=_copy_setup(entry.best_finish_rank_setup),
                    best_finish_time_ms=entry.best_finish_time_ms,
                    best_finish_time_rank=entry.best_finish_time_rank,
                    best_finish_time_setup=_copy_setup(entry.best_finish_time_setup),
                    latest_finish_rank=entry.latest_finish_rank,
                    latest_finish_time_ms=entry.latest_finish_time_ms,
                    latest_finish_delta_ms=entry.latest_finish_delta_ms,
                    latest_finish_setup=_copy_setup(entry.latest_finish_setup),
                    attempt_stats=entry.attempt_stats,
                    failed_attempt=entry.failed_attempt,
                )
                for key, entry in self.entries.items()
            },
            best_finish_position=self.best_finish_position,
        )

    def update(
        self,
        info: Mapping[str, object],
        telemetry: FZeroXTelemetry | None,
        *,
        episode_done: bool,
    ) -> TrackRecordBook:
        best_finish_position = _updated_best_finish_position(
            self.best_finish_position,
            info,
            telemetry,
        )
        track_key = track_record_key(info)
        if track_key is None:
            if best_finish_position == self.best_finish_position:
                return self
            return TrackRecordBook(
                entries={key: value for key, value in self.entries.items()},
                best_finish_position=best_finish_position,
            )
        current_entry = self.entries.get(track_key, TrackRecordEntry())
        updated_entry = current_entry.update(
            info=info,
            telemetry=telemetry,
            episode_done=episode_done,
        )
        if updated_entry == current_entry and best_finish_position == self.best_finish_position:
            return self
        entries = {key: value for key, value in self.entries.items()}
        entries[track_key] = updated_entry
        return TrackRecordBook(
            entries=entries,
            best_finish_position=best_finish_position,
        )

    def entry_for(self, info: Mapping[str, object]) -> TrackRecordEntry | None:
        for track_key in track_record_lookup_keys(info):
            entry = self.entries.get(track_key)
            if entry is not None:
                return entry
        return None
