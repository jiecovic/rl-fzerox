# src/rl_fzerox/core/envs/engine/reset/tracks.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from random import Random, random

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.config.schema import (
    TrackRecordsConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


@dataclass(frozen=True, slots=True)
class TrackSamplingLimits:
    """Limits that keep deterministic balanced sampling cycles bounded."""

    max_balanced_cycle_slots: int = 128


TRACK_SAMPLING_LIMITS = TrackSamplingLimits()


@dataclass(frozen=True)
class SelectedTrack:
    """Reset-time track baseline selected for the current episode."""

    id: str
    display_name: str | None
    course_ref: str | None
    course_id: str | None
    course_name: str | None
    baseline_state_path: Path
    weight: float
    course_index: int | None
    mode: str | None
    vehicle: str | None
    vehicle_name: str | None
    engine_setting: str | None
    records: TrackRecordsConfig | None
    sampling_mode: str
    cycle_position: int | None = None

    def info(self) -> dict[str, object]:
        info = {
            "track_sampling_enabled": True,
            "track_sampling_mode": self.sampling_mode,
            "track_id": self.id,
            "track_display_name": self.display_name,
            "track_course_ref": self.course_ref,
            "track_course_id": self.course_id,
            "track_course_name": self.course_name,
            "track_baseline_state_path": str(self.baseline_state_path),
            "track_sampling_weight": self.weight,
            "track_course_index": self.course_index,
            "track_mode": self.mode,
            "track_vehicle": self.vehicle,
            "track_vehicle_name": self.vehicle_name,
            "track_engine_setting": self.engine_setting,
            "track_sampling_cycle_position": self.cycle_position,
        }
        if self.records is not None:
            info.update(self.records.info())
        return info


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


class TrackResetSelector:
    """Select reset baselines using either iid random or per-env balanced cycling."""

    def __init__(self, *, env_index: int = 0) -> None:
        self._env_index = max(0, int(env_index))
        self._fingerprint: tuple[object, ...] | None = None
        self._cycle: tuple[TrackSamplingEntryConfig, ...] = ()
        self._cursor = 0

    def select(self, config: TrackSamplingConfig, *, seed: int | None) -> SelectedTrack | None:
        if config.sampling_mode == "random":
            return select_reset_track(config, seed=seed)
        if config.sampling_mode in ("balanced", "step_balanced"):
            return self._select_balanced(config, sampling_mode=config.sampling_mode)
        raise ValueError(f"Unsupported track sampling mode: {config.sampling_mode!r}")

    def select_sequential(self, config: TrackSamplingConfig) -> SelectedTrack | None:
        """Select configured tracks in list order, ignoring training weights."""

        if not config.enabled:
            return None
        if not config.entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_sequential_cycle(config)
        position = self._cursor % len(self._cycle)
        entry = self._cycle[position]
        self._cursor += 1
        return _selected_track_from_entry(
            entry,
            sampling_mode="sequential",
            cycle_position=position,
        )

    def _select_balanced(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        if not config.entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_balanced_cycle(config)
        position = self._cursor % len(self._cycle)
        entry = self._cycle[position]
        self._cursor += 1
        return _selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
        )

    def _sync_balanced_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = _track_sampling_fingerprint(config)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._cycle = _balanced_cycle(config.entries)
        self._cursor = self._env_index % len(self._cycle)

    def _sync_sequential_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = ("sequential", *_track_sampling_fingerprint(config))
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._cycle = tuple(config.entries)
        self._cursor = 0


def select_reset_track_by_course_id(
    config: TrackSamplingConfig,
    *,
    course_id: str,
) -> SelectedTrack | None:
    """Select the first configured reset baseline for a specific course id."""

    if not config.enabled:
        return None
    for entry in config.entries:
        if entry.course_id == course_id:
            return _selected_track_from_entry(entry, sampling_mode="locked")
    return None


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
    return _selected_track_from_entry(
        _weighted_entry(config.entries, sample=sample),
        sampling_mode="random",
    )


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


def _selected_track_from_entry(
    entry: TrackSamplingEntryConfig,
    *,
    sampling_mode: str,
    cycle_position: int | None = None,
) -> SelectedTrack:
    if entry.baseline_state_path is None:
        raise ValueError(
            f"track sampling entry {entry.id!r} has no materialized baseline_state_path"
        )
    return SelectedTrack(
        id=entry.id,
        display_name=entry.display_name,
        course_ref=entry.course_ref,
        course_id=entry.course_id,
        course_name=entry.course_name,
        baseline_state_path=entry.baseline_state_path,
        weight=float(entry.weight),
        course_index=None if entry.course_index is None else int(entry.course_index),
        mode=entry.mode,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        engine_setting=entry.engine_setting,
        records=entry.records,
        sampling_mode=sampling_mode,
        cycle_position=cycle_position,
    )


def _balanced_cycle(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    counts = _balanced_repetition_counts(entries)
    total = sum(counts)
    current = [0] * len(entries)
    cycle: list[TrackSamplingEntryConfig] = []
    for _ in range(total):
        for index, count in enumerate(counts):
            current[index] += count
        selected_index = max(range(len(entries)), key=lambda index: (current[index], -index))
        current[selected_index] -= total
        cycle.append(entries[selected_index])
    return tuple(cycle)


def _balanced_repetition_counts(entries: tuple[TrackSamplingEntryConfig, ...]) -> tuple[int, ...]:
    weights = [Fraction(str(float(entry.weight))).limit_denominator(100) for entry in entries]
    denominator = 1
    for weight in weights:
        denominator = denominator * weight.denominator // gcd(denominator, weight.denominator)
    counts = [max(1, int(weight * denominator)) for weight in weights]
    common_divisor = reduce(gcd, counts)
    counts = [count // common_divisor for count in counts]
    total = sum(counts)
    if total > TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots:
        scale = TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots / total
        counts = [max(1, round(count * scale)) for count in counts]
    return tuple(counts)


def _track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return (
        config.sampling_mode,
        tuple(
            (
                entry.id,
                entry.display_name,
                entry.course_ref,
                entry.course_id,
                entry.course_name,
                entry.baseline_state_path,
                float(entry.weight),
                entry.course_index,
                entry.mode,
                entry.vehicle,
                entry.vehicle_name,
                entry.engine_setting,
            )
            for entry in config.entries
        ),
    )
