# src/rl_fzerox/core/envs/engine/reset/track_sampling/selection.py
from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from functools import reduce
from math import gcd
from random import Random, choice, random
from typing import TypeVar

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.engine_tuning import (
    EngineTuningChoice,
    EngineTuningContext,
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.models import (
    TRACK_SAMPLING_LIMITS,
    SelectedTrack,
)
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_reset_target_key,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import EngineSetting, resolve_engine_setting

_T = TypeVar("_T")


class TrackResetSelector:
    """Select reset baselines using either iid random or per-env balanced cycling."""

    def __init__(self, *, env_index: int = 0) -> None:
        self._env_index = max(0, int(env_index))
        self._fingerprint: tuple[object, ...] | None = None
        self._cycle: tuple[TrackSamplingEntryConfig, ...] = ()
        self._sequential_course_buckets: tuple[tuple[TrackSamplingEntryConfig, ...], ...] = ()
        self._cursor = 0

    def select(
        self,
        config: TrackSamplingConfig,
        *,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None = None,
        engine_tuning_selection: EngineTuningSelectionMode = "sample",
    ) -> SelectedTrack | None:
        if config.sampling_mode == "random":
            return select_reset_track(
                config,
                seed=seed,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        if config.sampling_mode == "balanced":
            return self._select_balanced(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        if config.sampling_mode in {"step_balanced", "adaptive_step_balanced"}:
            return select_reset_track_by_course_weight(
                config,
                seed=seed,
                sampling_mode=config.sampling_mode,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        if config.sampling_mode in {"deficit_budget", "fixed_env"}:
            return self._select_fixed_env(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        raise ValueError(f"Unsupported track sampling mode: {config.sampling_mode!r}")

    def select_sequential(
        self,
        config: TrackSamplingConfig,
        *,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None = None,
        engine_tuning_selection: EngineTuningSelectionMode = "sample",
    ) -> SelectedTrack | None:
        """Select configured tracks in list order, ignoring training weights."""

        if not config.enabled:
            return None
        selectable_entries = _selectable_entries(config.entries)
        if not selectable_entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_sequential_cycle(config.model_copy(update={"entries": selectable_entries}))
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
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
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
            if any(_entry_matches_course_request(entry, course_id) for entry in bucket):
                self._cursor = index
                return True
        return False

    def _select_balanced(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None,
        engine_tuning_selection: EngineTuningSelectionMode,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        selectable_entries = _selectable_entries(config.entries)
        if not selectable_entries:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_balanced_cycle(config.model_copy(update={"entries": selectable_entries}))
        position = self._cursor % len(self._cycle)
        entry = self._cycle[position]
        self._cursor += 1
        return _selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )

    def _sync_balanced_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = _track_sampling_fingerprint(config)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._cycle = _balanced_cycle(config.entries)
        self._cursor = self._env_index % len(self._cycle)

    def _sync_sequential_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = ("sequential", *_sequential_track_sampling_fingerprint(config))
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._sequential_course_buckets = _sequential_course_buckets(config.entries)
        self._cursor = 0

    def _select_fixed_env(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None,
        engine_tuning_selection: EngineTuningSelectionMode,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        selectable_entries = _selectable_entries(config.entries)
        if not selectable_entries:
            raise ValueError("track sampling is enabled but has no entries")
        course_buckets = tuple(
            entries for _, entries in _group_entries_by_course(selectable_entries)
        )
        position = self._env_index % len(course_buckets)
        entry = _pick_track_entry(course_buckets[position], seed=seed)
        return _selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )


def select_reset_track_by_course_id(
    config: TrackSamplingConfig,
    *,
    course_id: str,
    sampling_mode: str = "locked",
    seed: int | None = None,
    engine_tuning_sampler: EngineTuningResetSampler | None = None,
    engine_tuning_selection: EngineTuningSelectionMode = "sample",
) -> SelectedTrack | None:
    """Select one configured reset baseline for a specific course id."""

    if not config.enabled:
        return None
    matching_entries = tuple(
        entry
        for entry in _selectable_entries(config.entries)
        if _entry_matches_course_request(entry, course_id)
    )
    if matching_entries:
        return _selected_track_from_entry(
            _pick_track_entry(matching_entries, seed=seed),
            sampling_mode=sampling_mode,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )
    return None


def select_reset_track(
    config: TrackSamplingConfig,
    *,
    seed: int | None,
    sampling_mode: str = "random",
    engine_tuning_sampler: EngineTuningResetSampler | None = None,
    engine_tuning_selection: EngineTuningSelectionMode = "sample",
) -> SelectedTrack | None:
    """Select one configured reset baseline with deterministic seeding when available."""

    if not config.enabled:
        return None
    entries = _selectable_entries(config.entries)
    if not entries:
        raise ValueError("track sampling is enabled but has no entries")

    total_weight = sum(float(entry.weight) for entry in entries)
    if total_weight <= 0.0:
        raise ValueError("track sampling requires at least one positive entry weight")
    sample = (Random(seed).random() if seed is not None else random()) * total_weight
    return _selected_track_from_entry(
        _weighted_entry(entries, sample=sample),
        sampling_mode=sampling_mode,
        seed=seed,
        engine_tuning_config=config.engine_tuning,
        engine_tuning_sampler=engine_tuning_sampler,
        engine_tuning_selection=engine_tuning_selection,
    )


def select_reset_track_by_course_weight(
    config: TrackSamplingConfig,
    *,
    seed: int | None,
    sampling_mode: str,
    engine_tuning_sampler: EngineTuningResetSampler | None = None,
    engine_tuning_selection: EngineTuningSelectionMode = "sample",
) -> SelectedTrack | None:
    """Select one course by summed weight, then one baseline within that course."""

    if not config.enabled:
        return None
    entries = _selectable_entries(config.entries)
    if not entries:
        raise ValueError("track sampling is enabled but has no entries")

    course_buckets = _group_entries_by_course(entries)
    course_weights = tuple(_entries_weight(entries) for _, entries in course_buckets)
    total_course_weight = sum(course_weights)
    if total_course_weight <= 0.0:
        raise ValueError("track sampling requires at least one positive course weight")

    rng = Random(seed) if seed is not None else Random()
    course_index = _weighted_index(course_weights, sample=rng.random() * total_course_weight)
    _, course_entries = course_buckets[course_index]
    entry_weight = course_weights[course_index]
    entry = _weighted_entry(course_entries, sample=rng.random() * entry_weight)
    return _selected_track_from_entry(
        entry,
        sampling_mode=sampling_mode,
        seed=seed,
        engine_tuning_config=config.engine_tuning,
        engine_tuning_sampler=engine_tuning_sampler,
        engine_tuning_selection=engine_tuning_selection,
    )


def _weighted_entry(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    sample: float,
) -> TrackSamplingEntryConfig:
    return entries[_weighted_index(tuple(float(entry.weight) for entry in entries), sample=sample)]


def _weighted_index(weights: tuple[float, ...], *, sample: float) -> int:
    cursor = 0.0
    for index, weight in enumerate(weights):
        cursor += max(0.0, float(weight))
        if sample < cursor:
            return index
    return len(weights) - 1


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


def _selectable_entries(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    return tuple(
        entry
        for entry in entries
        if not (
            entry.alt_baseline_id is not None
            and entry.baseline_state_path is not None
            and not entry.baseline_state_path.expanduser().is_file()
        )
    )


def _selected_track_from_entry(
    entry: TrackSamplingEntryConfig,
    *,
    sampling_mode: str,
    cycle_position: int | None = None,
    seed: int | None,
    engine_tuning_config: AdaptiveEngineTuningConfig,
    engine_tuning_sampler: EngineTuningResetSampler | None,
    engine_tuning_selection: EngineTuningSelectionMode,
) -> SelectedTrack:
    if entry.baseline_state_path is None:
        raise ValueError(
            f"track sampling entry {entry.id!r} has no materialized baseline_state_path"
        )
    engine_choice = _engine_tuning_choice(
        entry,
        config=engine_tuning_config,
        sampler=engine_tuning_sampler,
        seed=seed,
        selection=engine_tuning_selection,
    )
    resolved_engine = _resolved_engine_setting(entry, seed=seed, engine_choice=engine_choice)
    return SelectedTrack(
        id=entry.id,
        display_name=entry.display_name,
        course_ref=entry.course_ref,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_name=entry.course_name,
        baseline_state_path=entry.baseline_state_path,
        weight=float(entry.weight),
        course_index=None if entry.course_index is None else int(entry.course_index),
        mode=entry.mode,
        gp_difficulty=entry.gp_difficulty,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        engine_setting_raw_value=(
            None if resolved_engine is None else int(resolved_engine.raw_value)
        ),
        engine_setting_min_raw_value=entry.engine_setting_min_raw_value,
        engine_setting_max_raw_value=entry.engine_setting_max_raw_value,
        engine_tuning_context_key=None if engine_choice is None else engine_choice.context.key,
        engine_tuning_course_key=(
            None if engine_choice is None else engine_choice.context.course_key
        ),
        engine_tuning_vehicle_id=(
            None if engine_choice is None else engine_choice.context.vehicle_id
        ),
        engine_tuning_sampled_score=None if engine_choice is None else engine_choice.sampled_score,
        engine_tuning_mean_score=None if engine_choice is None else engine_choice.mean_score,
        engine_tuning_finish_count=None if engine_choice is None else engine_choice.finish_count,
        source_vehicle=entry.source_vehicle,
        source_course_index=entry.source_course_index,
        source_gp_difficulty=entry.source_gp_difficulty,
        source_engine_setting_raw_value=entry.source_engine_setting_raw_value,
        baseline_group_id=entry.baseline_group_id,
        baseline_group_weight=(
            None if entry.baseline_group_weight is None else float(entry.baseline_group_weight)
        ),
        alt_baseline_id=entry.alt_baseline_id,
        alt_baseline_label=entry.alt_baseline_label,
        alt_baseline_source_entry_id=entry.alt_baseline_source_entry_id,
        generated_course_kind=entry.generated_course_kind,
        generated_course_seed=entry.generated_course_seed,
        generated_course_hash=entry.generated_course_hash,
        generated_course_slot=entry.generated_course_slot,
        generated_course_generation=entry.generated_course_generation,
        generated_course_segment_count=entry.generated_course_segment_count,
        generated_course_length=entry.generated_course_length,
        log_per_course=entry.log_per_course,
        records=entry.records,
        sampling_mode=sampling_mode,
        cycle_position=cycle_position,
    )


def _resolved_engine_setting(
    entry: TrackSamplingEntryConfig,
    *,
    seed: int | None,
    engine_choice: EngineTuningChoice | None,
) -> EngineSetting | None:
    raw_value = (
        engine_choice.engine_setting_raw_value
        if engine_choice is not None
        else _target_engine_setting_raw_value(entry, seed=seed)
    )
    if raw_value is None:
        return None
    return resolve_engine_setting(
        raw_value,
        context=f"track sampling entry {entry.id!r}",
    )


def _engine_tuning_choice(
    entry: TrackSamplingEntryConfig,
    *,
    config: AdaptiveEngineTuningConfig,
    sampler: EngineTuningResetSampler | None,
    seed: int | None,
    selection: EngineTuningSelectionMode,
) -> EngineTuningChoice | None:
    if not config.enabled or sampler is None:
        return None
    return sampler.choose(
        engine_tuning_context_for_entry(entry),
        selection=selection,
        seed=seed,
    )


def engine_tuning_context_for_entry(entry: TrackSamplingEntryConfig) -> EngineTuningContext:
    """Return the adaptive engine-tuning context for one materialized reset entry."""

    return EngineTuningContext(
        course_key=_engine_tuning_course_key(entry),
        vehicle_id=entry.vehicle or entry.source_vehicle or "unknown",
    )


def _engine_tuning_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.generated_course_kind == X_CUP_COURSE.generated_kind:
        return "x_cup"
    return (
        entry.runtime_course_key
        or entry.course_id
        or entry.course_ref
        or (f"course_index:{entry.course_index}" if entry.course_index is not None else entry.id)
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
    raw_weights = tuple(max(0.0, float(weight)) for weight in weights)
    if not raw_weights:
        return ()
    if all(weight <= 0.0 for weight in raw_weights):
        return tuple(1 for _ in raw_weights)

    fractions = [
        Fraction(0) if weight <= 0.0 else Fraction(str(weight)).limit_denominator(100)
        for weight in raw_weights
    ]
    denominator = 1
    for weight in fractions:
        denominator = denominator * weight.denominator // gcd(denominator, weight.denominator)
    counts = [int(weight * denominator) for weight in fractions]
    positive_counts = [count for count in counts if count > 0]
    if not positive_counts:
        return tuple(1 for _ in raw_weights)
    common_divisor = reduce(gcd, positive_counts)
    counts = [count // common_divisor if count > 0 else 0 for count in counts]
    total = sum(counts)
    if total > TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots:
        scale = TRACK_SAMPLING_LIMITS.max_balanced_cycle_slots / total
        counts = [max(1, round(count * scale)) if count > 0 else 0 for count in counts]
    return tuple(counts)


def _track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return (
        config.sampling_mode,
        tuple(_entry_fingerprint(entry, include_weight=True) for entry in config.entries),
    )


def _sequential_track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return tuple(
        (
            target_key,
            tuple(_entry_fingerprint(entry, include_weight=False) for entry in entries),
        )
        for target_key, entries in _group_entries_by_sequential_target(config.entries)
    )


def _entry_fingerprint(
    entry: TrackSamplingEntryConfig,
    *,
    include_weight: bool,
) -> tuple[object, ...]:
    return (
        entry.id,
        entry.display_name,
        entry.course_ref,
        entry.course_id,
        entry.runtime_course_key,
        entry.course_name,
        entry.baseline_state_path,
        float(entry.weight) if include_weight else None,
        entry.course_index,
        entry.mode,
        entry.gp_difficulty,
        entry.vehicle,
        entry.vehicle_name,
        entry.source_vehicle,
        entry.engine_setting_raw_value,
        entry.engine_setting_min_raw_value,
        entry.engine_setting_max_raw_value,
        entry.source_course_index,
        entry.source_gp_difficulty,
        entry.source_engine_setting_raw_value,
        entry.baseline_group_id,
        entry.baseline_group_weight,
        entry.alt_baseline_id,
        entry.alt_baseline_label,
        entry.alt_baseline_source_entry_id,
        entry.generated_course_kind,
        entry.generated_course_seed,
        entry.generated_course_hash,
        entry.generated_course_slot,
        entry.generated_course_generation,
        entry.generated_course_segment_count,
        entry.generated_course_length,
        entry.log_per_course,
        entry.records.model_dump(mode="json") if entry.records is not None else None,
    )


def _sequential_course_buckets(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[TrackSamplingEntryConfig, ...], ...]:
    return tuple(entries for _, entries in _group_entries_by_sequential_target(entries))


def _group_entries_by_sequential_target(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[str, tuple[TrackSamplingEntryConfig, ...]], ...]:
    grouped: dict[str, list[TrackSamplingEntryConfig]] = {}
    order: list[str] = []
    for entry in entries:
        target_key = _entry_sequential_target_key(entry)
        bucket = grouped.get(target_key)
        if bucket is None:
            bucket = []
            grouped[target_key] = bucket
            order.append(target_key)
        bucket.append(entry)
    return tuple((target_key, tuple(grouped[target_key])) for target_key in order)


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
    if entry.runtime_course_key:
        return f"runtime_course_key:{entry.runtime_course_key}"
    if entry.course_id:
        return f"course_id:{entry.course_id}"
    if entry.course_ref:
        return f"course_ref:{entry.course_ref}"
    if entry.course_index is not None:
        return f"course_index:{int(entry.course_index)}"
    return f"entry:{entry.id}"


def _entry_matches_course_request(entry: TrackSamplingEntryConfig, course_id: str) -> bool:
    return course_id in (
        entry.runtime_course_key,
        entry.course_id,
        entry.id,
        _entry_sequential_target_key(entry),
    )


def _entry_sequential_target_key(entry: TrackSamplingEntryConfig) -> str:
    return track_sampling_reset_target_key(
        entry_id=entry.id,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_ref=entry.course_ref,
        course_index=entry.course_index,
        gp_difficulty=entry.gp_difficulty,
    )


def _entries_weight(entries: tuple[TrackSamplingEntryConfig, ...]) -> float:
    return sum(max(0.0, float(entry.weight)) for entry in entries)
