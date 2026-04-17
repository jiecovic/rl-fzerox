# src/rl_fzerox/core/envs/engine/tracks.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random, random

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.config.schema import TrackSamplingConfig, TrackSamplingEntryConfig


@dataclass(frozen=True)
class SelectedTrack:
    """Reset-time track baseline selected for the current episode."""

    id: str
    display_name: str | None
    baseline_state_path: Path
    weight: float
    course_index: int | None

    def info(self) -> dict[str, object]:
        return {
            "track_sampling_enabled": True,
            "track_id": self.id,
            "track_display_name": self.display_name,
            "track_baseline_state_path": str(self.baseline_state_path),
            "track_sampling_weight": self.weight,
            "track_course_index": self.course_index,
        }


class TrackBaselineCache:
    """Per-env cache of sampled track savestates.

    Subprocess workers keep their own cache, which avoids rereading the same
    multi-megabyte `.state` files on every episode reset.
    """

    def __init__(self) -> None:
        self._states_by_path: dict[Path, bytes] = {}

    def load_into_backend(self, backend: EmulatorBackend, path: Path) -> None:
        state = self._states_by_path.get(path)
        if state is None:
            state = path.read_bytes()
            self._states_by_path[path] = state
        backend.load_baseline_bytes(state, source_path=path)


def select_reset_track(
    config: TrackSamplingConfig,
    *,
    seed: int | None,
) -> SelectedTrack | None:
    """Select one configured reset baseline with deterministic seeding when available."""

    if not config.enabled:
        return None
    if not config.entries:
        raise ValueError("track sampling is enabled but has no entries")

    total_weight = sum(float(entry.weight) for entry in config.entries)
    sample = (Random(seed).random() if seed is not None else random()) * total_weight
    return _selected_track_from_entry(_weighted_entry(config.entries, sample=sample))


def _weighted_entry(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    sample: float,
) -> TrackSamplingEntryConfig:
    cursor = 0.0
    for entry in entries:
        cursor += float(entry.weight)
        if sample < cursor:
            return entry
    return entries[-1]


def _selected_track_from_entry(entry: TrackSamplingEntryConfig) -> SelectedTrack:
    return SelectedTrack(
        id=entry.id,
        display_name=entry.display_name,
        baseline_state_path=entry.baseline_state_path,
        weight=float(entry.weight),
        course_index=None if entry.course_index is None else int(entry.course_index),
    )
