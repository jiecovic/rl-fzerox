# src/rl_fzerox/core/envs/engine/reset/track_sampling/selection.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from random import Random, choice, random
from typing import TypeVar

from rl_fzerox.core.envs.engine.reset.track_sampling.models import (
    TRACK_SAMPLING_LIMITS,
    SelectedTrack,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig
from rl_fzerox.core.runtime_spec.vehicle_catalog import EngineSetting, resolve_engine_setting

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _WeightedCourseBucket:
    key: str
    weight: float
    entry_cycle: tuple[TrackSamplingEntryConfig, ...]


class TrackResetSelector:
    """Select reset baselines using either iid random or per-env balanced cycling."""

    def __init__(self, *, env_index: int = 0) -> None:
        self._env_index = max(0, int(env_index))
        self._fingerprint: tuple[object, ...] | None = None
        self._cycle: tuple[TrackSamplingEntryConfig, ...] = ()
        self._weighted_course_cycle: tuple[_WeightedCourseBucket, ...] = ()
        self._entry_cursors_by_course: dict[str, int] = {}
        self._sequential_course_buckets: tuple[tuple[TrackSamplingEntryConfig, ...], ...] = ()
        self._cursor = 0

    def select(self, config: TrackSamplingConfig, *, seed: int | None) -> SelectedTrack | None:
        if config.sampling_mode == "random":
            return select_reset_track(config, seed=seed)
        if config.sampling_mode == "balanced":
            return self._select_balanced(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
            )
        if config.sampling_mode in {"step_balanced", "adaptive_step_balanced"}:
            return self._select_weighted_course_balanced(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
            )
        raise ValueError(f"Unsupported track sampling mode: {config.sampling_mode!r}")

    def select_sequential(
        self,
        config: TrackSamplingConfig,
        *,
        seed: int | None,
    ) -> SelectedTrack | None:
        """Select configured tracks in list order, ignoring training weights."""

        if not config.enabled:
            return None
        if not config.entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_sequential_cycle(config)
        position = self._cursor % len(self._sequential_course_buckets)
        entry = _pick_track_entry(
            self._sequential_course_buckets[position],
            seed=seed,
        )
        self._cursor += 1
        return _selected_track_from_entry(
            entry,
            sampling_mode="sequential",
            cycle_position=position,
            seed=seed,
        )

    def set_next_sequential_course(
        self,
        config: TrackSamplingConfig,
        *,
        course_id: str,
    ) -> bool:
        """Align the next sequential watch reset to the requested course."""

        if not config.enabled or not config.entries:
            return False
        self._sync_sequential_cycle(config)
        for index, bucket in enumerate(self._sequential_course_buckets):
            if any(entry.course_id == course_id for entry in bucket):
                self._cursor = index
                return True
        return False

    def _select_balanced(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
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
            seed=seed,
        )

    def _select_weighted_course_balanced(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        if not config.entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_weighted_course_cycle(config)
        position = self._cursor % len(self._weighted_course_cycle)
        bucket = self._weighted_course_cycle[position]
        entry = self._next_bucket_entry(bucket)
        self._cursor += 1
        return _selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
            seed=seed,
        )

    def _sync_balanced_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = _track_sampling_fingerprint(config)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._cycle = _balanced_cycle(config.entries)
        self._cursor = self._env_index % len(self._cycle)

    def _sync_weighted_course_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = ("weighted_course", *_track_sampling_fingerprint(config))
        if fingerprint == self._fingerprint:
            return
        previous_cycle = self._weighted_course_cycle
        previous_cursor = self._cursor
        previous_entry_cursors = self._entry_cursors_by_course
        self._fingerprint = fingerprint
        self._weighted_course_cycle = _weighted_course_cycle(config.entries)
        self._entry_cursors_by_course = {
            bucket.key: previous_entry_cursors.get(
                bucket.key,
                self._env_index % len(bucket.entry_cycle),
            )
            for bucket in self._weighted_course_cycle
        }
        self._cursor = (
            previous_cursor if previous_cycle else self._env_index
        ) % len(self._weighted_course_cycle)

    def _sync_sequential_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = ("sequential", *_sequential_track_sampling_fingerprint(config))
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._sequential_course_buckets = _sequential_course_buckets(config.entries)
        self._cursor = 0

    def _next_bucket_entry(self, bucket: _WeightedCourseBucket) -> TrackSamplingEntryConfig:
        cursor = self._entry_cursors_by_course.get(bucket.key, 0)
        entry = bucket.entry_cycle[cursor % len(bucket.entry_cycle)]
        self._entry_cursors_by_course[bucket.key] = cursor + 1
        return entry


def select_reset_track_by_course_id(
    config: TrackSamplingConfig,
    *,
    course_id: str,
    seed: int | None = None,
) -> SelectedTrack | None:
    """Select one configured reset baseline for a specific course id."""

    if not config.enabled:
        return None
    matching_entries = tuple(entry for entry in config.entries if entry.course_id == course_id)
    if matching_entries:
        return _selected_track_from_entry(
            _pick_track_entry(matching_entries, seed=seed),
            sampling_mode="locked",
            seed=seed,
        )
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
        seed=seed,
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


def _pick_track_entry(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    seed: int | None,
) -> TrackSamplingEntryConfig:
    if len(entries) == 1:
        return entries[0]
    if seed is not None:
        return Random(seed).choice(entries)
    return choice(entries)


def _selected_track_from_entry(
    entry: TrackSamplingEntryConfig,
    *,
    sampling_mode: str,
    cycle_position: int | None = None,
    seed: int | None,
) -> SelectedTrack:
    if entry.baseline_state_path is None:
        raise ValueError(
            f"track sampling entry {entry.id!r} has no materialized baseline_state_path"
        )
    resolved_engine = _resolved_engine_setting(entry, seed=seed)
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
        gp_difficulty=entry.gp_difficulty,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        engine_setting=resolved_engine.id if resolved_engine is not None else entry.engine_setting,
        engine_setting_raw_value=(
            None if resolved_engine is None else int(resolved_engine.raw_value)
        ),
        engine_setting_min_raw_value=entry.engine_setting_min_raw_value,
        engine_setting_max_raw_value=entry.engine_setting_max_raw_value,
        source_vehicle=entry.source_vehicle,
        source_course_index=entry.source_course_index,
        source_gp_difficulty=entry.source_gp_difficulty,
        source_engine_setting=entry.source_engine_setting,
        source_engine_setting_raw_value=entry.source_engine_setting_raw_value,
        records=entry.records,
        sampling_mode=sampling_mode,
        cycle_position=cycle_position,
    )


def _resolved_engine_setting(
    entry: TrackSamplingEntryConfig,
    *,
    seed: int | None,
) -> EngineSetting | None:
    raw_value = _target_engine_setting_raw_value(entry, seed=seed)
    if raw_value is None:
        return None
    return resolve_engine_setting(
        raw_value,
        context=f"track sampling entry {entry.id!r}",
    )


def _target_engine_setting_raw_value(
    entry: TrackSamplingEntryConfig,
    *,
    seed: int | None,
) -> int | None:
    minimum = entry.engine_setting_min_raw_value
    maximum = entry.engine_setting_max_raw_value
    if minimum is None and maximum is None:
        return entry.engine_setting_raw_value
    if minimum is None or maximum is None:
        raise ValueError(f"track sampling entry {entry.id!r} must define both engine range bounds")
    if minimum > maximum:
        raise ValueError(
            f"track sampling entry {entry.id!r} has engine range min > max: {minimum} > {maximum}"
        )
    if minimum == maximum:
        return int(minimum)
    rng = Random(seed) if seed is not None else None
    return (
        rng.randint(int(minimum), int(maximum))
        if rng is not None
        else Random().randint(int(minimum), int(maximum))
    )


def _balanced_cycle(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    counts = _balanced_repetition_counts(float(entry.weight) for entry in entries)
    return _balanced_cycle_from_counts(entries, counts)


def _weighted_course_cycle(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[_WeightedCourseBucket, ...]:
    buckets = _weighted_course_buckets(entries)
    counts = _balanced_repetition_counts(bucket.weight for bucket in buckets)
    return _balanced_cycle_from_counts(buckets, counts)


def _balanced_cycle_from_counts(items: tuple[_T, ...], counts: tuple[int, ...]) -> tuple[_T, ...]:
    total = sum(counts)
    current = [0] * len(items)
    cycle: list[_T] = []
    for _ in range(total):
        for index, count in enumerate(counts):
            current[index] += count
        selected_index = max(range(len(items)), key=lambda index: (current[index], -index))
        current[selected_index] -= total
        cycle.append(items[selected_index])
    return tuple(cycle)


def _balanced_repetition_counts(weights: Iterable[float]) -> tuple[int, ...]:
    fractions = [Fraction(str(float(weight))).limit_denominator(100) for weight in weights]
    denominator = 1
    for weight in fractions:
        denominator = denominator * weight.denominator // gcd(denominator, weight.denominator)
    counts = [max(1, int(weight * denominator)) for weight in fractions]
    common_divisor = reduce(gcd, counts)
    counts = [count // common_divisor for count in counts]
    total = sum(counts)
    if total > TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots:
        scale = TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots / total
        counts = [max(1, round(count * scale)) for count in counts]
    return tuple(counts)


def _weighted_course_buckets(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[_WeightedCourseBucket, ...]:
    return tuple(
        _WeightedCourseBucket(
            key=course_key,
            weight=sum(float(entry.weight) for entry in course_entries),
            entry_cycle=_balanced_cycle(course_entries),
        )
        for course_key, course_entries in _group_entries_by_course(entries)
    )


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


def _sequential_track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return tuple(
        (
            course_key,
            tuple(
                (
                    entry.id,
                    entry.baseline_state_path,
                    entry.vehicle,
                    entry.engine_setting,
                    entry.engine_setting_raw_value,
                    entry.engine_setting_min_raw_value,
                    entry.engine_setting_max_raw_value,
                )
                for entry in entries
            ),
        )
        for course_key, entries in _group_entries_by_course(config.entries)
    )


def _sequential_course_buckets(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[TrackSamplingEntryConfig, ...], ...]:
    return tuple(entries for _, entries in _group_entries_by_course(entries))


def _group_entries_by_course(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[str, tuple[TrackSamplingEntryConfig, ...]], ...]:
    grouped: dict[str, list[TrackSamplingEntryConfig]] = {}
    order: list[str] = []
    for entry in entries:
        course_key = _entry_course_key(entry)
        bucket = grouped.get(course_key)
        if bucket is None:
            bucket = []
            grouped[course_key] = bucket
            order.append(course_key)
        bucket.append(entry)
    return tuple((course_key, tuple(grouped[course_key])) for course_key in order)


def _entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.course_id:
        return f"course_id:{entry.course_id}"
    if entry.course_ref:
        return f"course_ref:{entry.course_ref}"
    if entry.course_index is not None:
        return f"course_index:{int(entry.course_index)}"
    return f"entry:{entry.id}"
