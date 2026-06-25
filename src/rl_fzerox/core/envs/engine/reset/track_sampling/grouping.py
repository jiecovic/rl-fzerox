# src/rl_fzerox/core/envs/engine/reset/track_sampling/grouping.py
"""Grouping helpers for track-sampling entries.

Deficit-budget and balanced sampling need stable reset-target groupings so
variants and generated-course slots roll up to the intended course target.
"""

from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_reset_target_key,
)


def track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return (
        config.sampling_mode,
        tuple(entry_fingerprint(entry, include_weight=True) for entry in config.entries),
    )


def sequential_track_sampling_fingerprint(config: TrackSamplingConfig) -> tuple[object, ...]:
    return tuple(
        (
            target_key,
            tuple(entry_fingerprint(entry, include_weight=False) for entry in entries),
        )
        for target_key, entries in group_entries_by_sequential_target(config.entries)
    )


def entry_fingerprint(
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
        entry.baseline_variant_index,
        entry.baseline_variant_count,
        entry.baseline_variant_seed,
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


def sequential_course_buckets(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[TrackSamplingEntryConfig, ...], ...]:
    return tuple(entries for _, entries in group_entries_by_sequential_target(entries))


def group_entries_by_sequential_target(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[str, tuple[TrackSamplingEntryConfig, ...]], ...]:
    grouped: dict[str, list[TrackSamplingEntryConfig]] = {}
    order: list[str] = []
    for entry in entries:
        target_key = entry_sequential_target_key(entry)
        bucket = grouped.get(target_key)
        if bucket is None:
            bucket = []
            grouped[target_key] = bucket
            order.append(target_key)
        bucket.append(entry)
    return tuple((target_key, tuple(grouped[target_key])) for target_key in order)


def group_entries_by_course(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[tuple[str, tuple[TrackSamplingEntryConfig, ...]], ...]:
    grouped: dict[str, list[TrackSamplingEntryConfig]] = {}
    order: list[str] = []
    for entry in entries:
        course_key = entry_course_key(entry)
        bucket = grouped.get(course_key)
        if bucket is None:
            bucket = []
            grouped[course_key] = bucket
            order.append(course_key)
        bucket.append(entry)
    return tuple((course_key, tuple(grouped[course_key])) for course_key in order)


def entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.runtime_course_key:
        return f"runtime_course_key:{entry.runtime_course_key}"
    if entry.course_id:
        return f"course_id:{entry.course_id}"
    if entry.course_ref:
        return f"course_ref:{entry.course_ref}"
    if entry.course_index is not None:
        return f"course_index:{int(entry.course_index)}"
    return f"entry:{entry.id}"


def entry_matches_course_request(entry: TrackSamplingEntryConfig, course_id: str) -> bool:
    return course_id in (
        entry.runtime_course_key,
        entry.course_id,
        entry.id,
        entry_sequential_target_key(entry),
    )


def entry_sequential_target_key(entry: TrackSamplingEntryConfig) -> str:
    return track_sampling_reset_target_key(
        entry_id=entry.id,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_ref=entry.course_ref,
        course_index=entry.course_index,
        gp_difficulty=entry.gp_difficulty,
    )


def entries_weight(entries: tuple[TrackSamplingEntryConfig, ...]) -> float:
    return sum(max(0.0, float(entry.weight)) for entry in entries)
