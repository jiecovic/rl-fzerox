# src/rl_fzerox/core/envs/engine/reset/track_sampling/modes.py
"""Sampling-mode algorithms for reset target entries.

This module builds deterministic weighted cycles and per-env assignments used by
equal, step-balanced, fixed-env, and deficit-budget sampling.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from functools import reduce
from math import gcd
from random import Random, choice
from typing import TypeVar

from rl_fzerox.core.envs.engine.reset.track_sampling.grouping import (
    entries_weight,
    group_entries_by_course,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.models import TRACK_SAMPLING_LIMITS
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig

_T = TypeVar("_T")


def weighted_course_entry(
    config: TrackSamplingConfig,
    *,
    seed: int | None,
) -> TrackSamplingEntryConfig | None:
    if not config.enabled:
        return None
    entries = selectable_entries(config.entries)
    if not entries:
        raise ValueError("track sampling is enabled but has no entries")

    course_buckets = group_entries_by_course(entries)
    course_weights = tuple(entries_weight(entries) for _, entries in course_buckets)
    total_course_weight = sum(course_weights)
    if total_course_weight <= 0.0:
        raise ValueError("track sampling requires at least one positive course weight")

    rng = Random(seed) if seed is not None else Random()
    course_index = weighted_index(course_weights, sample=rng.random() * total_course_weight)
    _, course_entries = course_buckets[course_index]
    entry_weight = course_weights[course_index]
    return weighted_entry(course_entries, sample=rng.random() * entry_weight)


def weighted_entry(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    sample: float,
) -> TrackSamplingEntryConfig:
    return entries[weighted_index(tuple(float(entry.weight) for entry in entries), sample=sample)]


def weighted_index(weights: tuple[float, ...], *, sample: float) -> int:
    cursor = 0.0
    for index, weight in enumerate(weights):
        cursor += max(0.0, float(weight))
        if sample < cursor:
            return index
    return len(weights) - 1


def pick_track_entry(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    seed: int | None,
) -> TrackSamplingEntryConfig:
    if len(entries) == 1:
        return entries[0]
    if seed is not None:
        return Random(seed).choice(entries)
    return choice(entries)


def selectable_entries(
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


def balanced_cycle(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    counts = balanced_repetition_counts(float(entry.weight) for entry in entries)
    return balanced_cycle_from_counts(entries, counts)


def balanced_cycle_from_counts(items: tuple[_T, ...], counts: tuple[int, ...]) -> tuple[_T, ...]:
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


def balanced_repetition_counts(weights: Iterable[float]) -> tuple[int, ...]:
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
