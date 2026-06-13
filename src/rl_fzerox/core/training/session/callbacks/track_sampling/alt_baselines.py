# src/rl_fzerox/core/training/session/callbacks/track_sampling/alt_baselines.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    reset_variant_key,
    track_sampling_artifact_course_key,
    track_sampling_artifact_reset_variant_key,
)


@dataclass(frozen=True, slots=True)
class TrackSamplingAltBaseline:
    """One manager-owned extra reset candidate for a stable track context."""

    id: str
    run_id: str
    course_key: str
    reset_variant_key: str
    source_entry_id: str
    label: str
    state_path: Path
    weight: float
    enabled: bool
    created_at: str
    updated_at: str
    deleted_at: str | None = None

    @property
    def active(self) -> bool:
        return self.enabled and self.deleted_at is None and self.state_path.is_file()


def apply_alt_baselines_to_track_sampling(
    config: TrackSamplingConfig,
    baselines: tuple[TrackSamplingAltBaseline, ...],
) -> TrackSamplingConfig:
    """Return a config with active stable-course alt baselines added as reset candidates."""

    base_entries = _base_entries_with_original_weights(config.entries)
    if not config.enabled or not base_entries:
        return config.model_copy(update={"entries": base_entries})

    baselines_by_key = _active_baselines_by_key(baselines)
    entries: list[TrackSamplingEntryConfig] = []
    for entry in base_entries:
        if entry.generated_course_kind == X_CUP_COURSE.generated_kind:
            entries.append(_without_baseline_group(entry))
            continue
        entry_key = _entry_alt_baseline_key(entry)
        matching = tuple(
            baseline
            for baseline in baselines_by_key.get(entry_key, ())
            if baseline.source_entry_id == entry.id
        )
        if not matching:
            entries.append(_without_baseline_group(entry))
            continue
        entries.extend(_entry_with_alt_baselines(entry, matching))
    return config.model_copy(update={"entries": tuple(entries)})


def strip_alt_baselines(config: TrackSamplingConfig) -> TrackSamplingConfig:
    """Remove generated alt-baseline entries and reset base entry group metadata."""

    return config.model_copy(
        update={
            "entries": _base_entries_with_original_weights(config.entries),
        }
    )


def alt_baseline_signature(
    baselines: tuple[TrackSamplingAltBaseline, ...],
) -> tuple[tuple[object, ...], ...]:
    """Return a compact active-baseline signature for cheap refresh checks."""

    return tuple(
        (
            baseline.id,
            baseline.course_key,
            baseline.reset_variant_key,
            baseline.source_entry_id,
            str(baseline.state_path),
            float(baseline.weight),
            baseline.updated_at,
        )
        for baseline in baselines
        if _is_selectable_alt_baseline(baseline)
    )


def alt_baseline_reset_variant_key(
    *,
    mode: str | None,
    gp_difficulty: str | None,
    vehicle: str | None,
) -> str:
    return reset_variant_key(mode=mode, gp_difficulty=gp_difficulty, vehicle=vehicle)


def stable_entry_alt_baseline_key(entry: TrackSamplingEntryConfig) -> tuple[str, str] | None:
    """Return the stable registry key for one entry, or None for generated X Cup entries."""

    if entry.generated_course_kind == X_CUP_COURSE.generated_kind:
        return None
    return _entry_alt_baseline_key(entry)


def _entry_alt_baseline_key(entry: TrackSamplingEntryConfig) -> tuple[str, str]:
    return (
        track_sampling_artifact_course_key(entry),
        track_sampling_artifact_reset_variant_key(entry),
    )


def _active_baselines_by_key(
    baselines: tuple[TrackSamplingAltBaseline, ...],
) -> dict[tuple[str, str], tuple[TrackSamplingAltBaseline, ...]]:
    grouped: dict[tuple[str, str], list[TrackSamplingAltBaseline]] = {}
    for baseline in baselines:
        if not _is_selectable_alt_baseline(baseline):
            continue
        key = (baseline.course_key, baseline.reset_variant_key)
        grouped.setdefault(key, []).append(baseline)
    return {key: tuple(items) for key, items in grouped.items()}


def _base_entries_with_original_weights(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    group_weights: dict[str, float] = {}
    for entry in entries:
        group_id = entry.baseline_group_id
        if group_id is None:
            continue
        group_weights[group_id] = group_weights.get(group_id, 0.0) + float(entry.weight)

    base_entries: list[TrackSamplingEntryConfig] = []
    for entry in entries:
        if entry.alt_baseline_id is not None:
            continue
        update: dict[str, object] = {
            "baseline_group_id": None,
            "baseline_group_weight": None,
            "alt_baseline_id": None,
            "alt_baseline_label": None,
            "alt_baseline_source_entry_id": None,
        }
        if entry.baseline_group_id == entry.id and entry.id in group_weights:
            update["weight"] = group_weights[entry.id]
        base_entries.append(entry.model_copy(update=update))
    return tuple(base_entries)


def _is_selectable_alt_baseline(baseline: TrackSamplingAltBaseline) -> bool:
    return baseline.active and baseline.weight > 0.0


def _entry_with_alt_baselines(
    entry: TrackSamplingEntryConfig,
    baselines: tuple[TrackSamplingAltBaseline, ...],
) -> tuple[TrackSamplingEntryConfig, ...]:
    ratio_total = 1.0 + sum(max(0.0, baseline.weight) for baseline in baselines)
    if ratio_total <= 0.0:
        return (entry,)
    base_weight = float(entry.weight)
    grouped_base = entry.model_copy(
        update={
            "baseline_group_id": entry.id,
            "baseline_group_weight": 1.0,
            "weight": base_weight / ratio_total,
        }
    )
    return (
        grouped_base,
        *(
            entry.model_copy(
                update={
                    "id": f"{entry.id}__alt_{baseline.id}",
                    "display_name": _alt_display_name(entry, baseline),
                    "baseline_state_path": baseline.state_path,
                    "weight": base_weight * max(0.0, baseline.weight) / ratio_total,
                    "baseline_group_id": entry.id,
                    "baseline_group_weight": max(0.0, baseline.weight),
                    "alt_baseline_id": baseline.id,
                    "alt_baseline_label": baseline.label,
                    "alt_baseline_source_entry_id": entry.id,
                    "log_per_course": entry.log_per_course,
                }
            )
            for baseline in baselines
        ),
    )


def _without_baseline_group(entry: TrackSamplingEntryConfig) -> TrackSamplingEntryConfig:
    if (
        entry.baseline_group_id is None
        and entry.baseline_group_weight is None
        and entry.alt_baseline_id is None
        and entry.alt_baseline_label is None
        and entry.alt_baseline_source_entry_id is None
    ):
        return entry
    return entry.model_copy(
        update={
            "baseline_group_id": None,
            "baseline_group_weight": None,
            "alt_baseline_id": None,
            "alt_baseline_label": None,
            "alt_baseline_source_entry_id": None,
        }
    )


def _alt_display_name(
    entry: TrackSamplingEntryConfig,
    baseline: TrackSamplingAltBaseline,
) -> str:
    base_label = entry.display_name or entry.course_name or entry.course_id or entry.id
    return f"{base_label} alt: {baseline.label}"
