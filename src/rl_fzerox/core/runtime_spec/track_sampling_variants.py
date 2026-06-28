# src/rl_fzerox/core/runtime_spec/track_sampling_variants.py
"""Baseline-variant expansion shared by materialization and runtime projection.

Training materialization, managed-run restore, and watch self-healing must agree
on how a single GP race entry fans out into opponent-grid variants. Keeping the
logic here prevents Watch from re-materializing entries that SQLite already has
as run-local artifacts.
"""

from __future__ import annotations

import hashlib

from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig


def expanded_baseline_variant_entries(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    baseline_variant_count: int,
) -> tuple[TrackSamplingEntryConfig, ...]:
    """Expand built-in GP race entries into materialized baseline variants."""

    if baseline_variant_count <= 1:
        return entries
    expanded: list[TrackSamplingEntryConfig] = []
    for entry in entries:
        if not entry_supports_baseline_variants(entry):
            expanded.append(entry)
            continue
        expanded.extend(
            baseline_variant_entry(
                entry,
                baseline_variant_index=variant_index,
                baseline_variant_count=baseline_variant_count,
            )
            for variant_index in range(baseline_variant_count)
        )
    return tuple(expanded)


def entry_supports_baseline_variants(entry: TrackSamplingEntryConfig) -> bool:
    """Return whether one spec entry owns generated GP opponent-grid variants."""

    return (
        entry.mode == "gp_race"
        and entry.baseline_variant_index is None
        and entry.alt_baseline_id is None
        and entry.generated_course_kind is None
        and entry.course_index is not None
        and entry.vehicle is not None
    )


def baseline_variant_entry(
    entry: TrackSamplingEntryConfig,
    *,
    baseline_variant_index: int,
    baseline_variant_count: int,
) -> TrackSamplingEntryConfig:
    """Return one materialized variant entry for a built-in GP race target."""

    update: dict[str, object] = {
        "weight": float(entry.weight) / baseline_variant_count,
        "baseline_group_id": entry.baseline_group_id or entry.id,
        "baseline_group_weight": 1.0,
        "baseline_variant_index": baseline_variant_index,
        "baseline_variant_count": baseline_variant_count,
    }
    if baseline_variant_index > 0:
        update["id"] = f"{entry.id}__variant_{baseline_variant_index + 1}"
        update["baseline_variant_seed"] = baseline_variant_seed(
            entry,
            baseline_variant_index=baseline_variant_index,
        )
    return entry.model_copy(update=update)


def baseline_variant_seed(
    entry: TrackSamplingEntryConfig,
    *,
    baseline_variant_index: int,
) -> int:
    """Derive stable reusable opponent-grid seeds without sharing RNG state."""

    parts = (
        "baseline_variant",
        entry.id,
        str(entry.course_id or ""),
        str(entry.runtime_course_key or ""),
        str(entry.course_index if entry.course_index is not None else ""),
        str(entry.gp_difficulty or ""),
        str(entry.vehicle or ""),
        str(baseline_variant_index),
    )
    digest = hashlib.blake2s("|".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
