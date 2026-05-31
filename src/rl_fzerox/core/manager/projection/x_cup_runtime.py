# src/rl_fzerox/core/manager/projection/x_cup_runtime.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import (
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


@dataclass(frozen=True, slots=True)
class GeneratedXCupRuntimeEntry:
    course_key: str
    slot: int
    entry_id: str
    course_id: str
    course_name: str
    baseline_state_path: Path
    generation: int | None
    course_hash: str | None
    course_seed: int | None
    segment_count: int | None
    course_length: float | None


def restore_generated_x_cup_entries_from_state(
    config: TrainAppConfig,
    *,
    state: TrackSamplingRuntimeState | None,
) -> TrainAppConfig:
    """Apply mutable generated X Cup slot identity from runtime state.

    Managed run config in SQLite remains the run spec source of truth. Generated
    X Cup baselines are runtime state because they rotate while the run trains.
    """

    track_sampling = restore_generated_x_cup_track_sampling_from_state(
        config.env.track_sampling,
        state=state,
    )
    if track_sampling is config.env.track_sampling:
        return config
    return config.model_copy(
        update={
            "env": config.env.model_copy(update={"track_sampling": track_sampling}),
        }
    )


def restore_generated_x_cup_track_sampling_from_state(
    config: TrackSamplingConfig,
    *,
    state: TrackSamplingRuntimeState | None,
) -> TrackSamplingConfig:
    """Apply mutable generated X Cup slot identity to one track-sampling config."""

    entries_by_key = _generated_entries_from_state(state)
    if not entries_by_key:
        return config
    next_entries: list[TrackSamplingEntryConfig] = []
    changed = False
    for entry in config.entries:
        runtime_entry = _runtime_entry_for_track_entry(entry, entries_by_key)
        if runtime_entry is None:
            next_entries.append(entry)
            continue
        next_entries.append(_restore_entry(entry, runtime_entry))
        changed = True
    if not changed:
        return config
    return config.model_copy(update={"entries": tuple(next_entries)})


def _generated_entries_from_state(
    state: TrackSamplingRuntimeState | None,
) -> dict[str, GeneratedXCupRuntimeEntry]:
    if state is None:
        return {}
    return {
        entry.course_key: runtime_entry
        for entry in state.entries
        if (runtime_entry := _generated_entry_from_runtime_entry(entry)) is not None
    }


def _generated_entry_from_runtime_entry(
    entry: TrackSamplingRuntimeEntry,
) -> GeneratedXCupRuntimeEntry | None:
    generated_course_slot = entry.generated_course_slot
    generated_entry_id = entry.generated_entry_id
    generated_course_id = entry.generated_course_id
    generated_course_name = entry.generated_course_name
    generated_baseline_state_path = entry.generated_baseline_state_path
    if (
        not isinstance(generated_course_slot, int)
        or not isinstance(generated_entry_id, str)
        or not isinstance(generated_course_id, str)
        or not isinstance(generated_course_name, str)
        or not isinstance(generated_baseline_state_path, str)
    ):
        return None
    return GeneratedXCupRuntimeEntry(
        course_key=entry.course_key,
        slot=max(0, generated_course_slot),
        entry_id=generated_entry_id,
        course_id=generated_course_id,
        course_name=generated_course_name,
        baseline_state_path=Path(generated_baseline_state_path).expanduser().resolve(),
        generation=entry.generated_course_generation,
        course_hash=entry.generated_course_hash,
        course_seed=entry.generated_course_seed,
        segment_count=entry.generated_course_segment_count,
        course_length=entry.generated_course_length,
    )


def _runtime_entry_for_track_entry(
    entry: TrackSamplingEntryConfig,
    entries_by_key: Mapping[str, GeneratedXCupRuntimeEntry],
) -> GeneratedXCupRuntimeEntry | None:
    if (
        entry.generated_course_kind != X_CUP_COURSE.generated_kind
        or entry.generated_course_slot is None
    ):
        return None
    key = entry.runtime_course_key
    if key is not None:
        runtime_entry = entries_by_key.get(key)
        if runtime_entry is not None:
            return runtime_entry
    return next(
        (
            runtime_entry
            for runtime_entry in entries_by_key.values()
            if runtime_entry.slot == entry.generated_course_slot
        ),
        None,
    )


def _restore_entry(
    entry: TrackSamplingEntryConfig,
    runtime_entry: GeneratedXCupRuntimeEntry,
) -> TrackSamplingEntryConfig:
    return entry.model_copy(
        update={
            "id": runtime_entry.entry_id,
            "runtime_course_key": runtime_entry.course_key,
            "course_id": runtime_entry.course_id,
            "course_name": runtime_entry.course_name,
            "display_name": runtime_entry.course_name,
            "baseline_state_path": runtime_entry.baseline_state_path,
            "generated_course_hash": runtime_entry.course_hash,
            "generated_course_seed": runtime_entry.course_seed,
            "generated_course_generation": runtime_entry.generation,
            "generated_course_segment_count": runtime_entry.segment_count,
            "generated_course_length": runtime_entry.course_length,
            "log_per_course": False,
        }
    )
