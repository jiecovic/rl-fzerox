# src/rl_fzerox/core/manager/projection/x_cup_runtime.py
"""X-Cup runtime restoration from SQLite-managed generated slots."""

from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import (
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
)
from rl_fzerox.core.runtime_spec.track_sampling_identity import track_sampling_entry_id
from rl_fzerox.core.runtime_spec.track_sampling_variants import (
    expanded_baseline_variant_entries,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
    track_sampling_artifact_course_key,
    track_sampling_artifact_index,
    track_sampling_artifact_reset_variant_key,
)


def restore_generated_x_cup_entries_from_slots(
    config: TrainAppConfig,
    *,
    slots: tuple[GeneratedXCupSlot, ...],
) -> TrainAppConfig:
    """Apply mutable generated X Cup slot identity from manager slot state.

    Managed run config in SQLite remains the run spec source of truth. Generated
    X Cup baselines are runtime state because they rotate while the run trains.
    """

    track_sampling = restore_generated_x_cup_track_sampling_from_slots(
        config.env.track_sampling,
        slots=slots,
    )
    if track_sampling is config.env.track_sampling:
        return config
    return config.model_copy(
        update={
            "env": config.env.model_copy(update={"track_sampling": track_sampling}),
        }
    )


def restore_generated_x_cup_track_sampling_from_slots(
    config: TrackSamplingConfig,
    *,
    slots: tuple[GeneratedXCupSlot, ...],
) -> TrackSamplingConfig:
    """Apply mutable generated X Cup slot identity to one track-sampling config."""

    slots_by_key = {slot.course_key: slot for slot in slots}
    slots_by_index = {slot.slot: slot for slot in slots}
    if not slots_by_key:
        return config
    next_entries: list[TrackSamplingEntryConfig] = []
    changed = False
    for entry in config.entries:
        slot = _slot_for_track_entry(entry, slots_by_key, slots_by_index)
        if slot is None:
            next_entries.append(entry)
            continue
        next_entries.append(_restore_entry(entry, slot))
        changed = True
    if not changed:
        return config
    return config.model_copy(update={"entries": tuple(next_entries)})


def restore_track_sampling_artifacts(
    config: TrainAppConfig,
    *,
    artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
) -> TrainAppConfig:
    """Apply manager-owned reset artifacts to one train config."""

    track_sampling = restore_track_sampling_config_artifacts(
        config.env.track_sampling,
        artifacts=artifacts,
    )
    if track_sampling is config.env.track_sampling:
        return config
    return config.model_copy(
        update={
            "env": config.env.model_copy(update={"track_sampling": track_sampling}),
        }
    )


def restore_track_sampling_config_artifacts(
    config: TrackSamplingConfig,
    *,
    artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
) -> TrackSamplingConfig:
    """Apply materialized reset artifact paths to track-sampling entries."""

    artifact_index = track_sampling_artifact_index(artifacts)
    if not artifact_index:
        return config
    restored_entries = expanded_baseline_variant_entries(
        config.entries,
        baseline_variant_count=config.baseline_variant_count,
    )
    next_entries: list[TrackSamplingEntryConfig] = []
    changed = restored_entries != config.entries
    for entry in restored_entries:
        artifact = artifact_index.get(
            (
                track_sampling_artifact_course_key(entry),
                track_sampling_artifact_reset_variant_key(entry),
            )
        )
        if artifact is None or not _artifact_matches_entry_identity(entry, artifact):
            next_entries.append(entry)
            continue
        next_entry = _restore_entry_artifact(entry, artifact)
        next_entries.append(next_entry)
        changed = changed or next_entry != entry
    if not changed:
        return config
    return config.model_copy(update={"entries": tuple(next_entries)})


def _slot_for_track_entry(
    entry: TrackSamplingEntryConfig,
    slots_by_key: Mapping[str, GeneratedXCupSlot],
    slots_by_index: Mapping[int, GeneratedXCupSlot],
) -> GeneratedXCupSlot | None:
    if (
        entry.generated_course_kind != X_CUP_COURSE.generated_kind
        or entry.generated_course_slot is None
    ):
        return None
    key = entry.runtime_course_key
    if key is not None:
        slot = slots_by_key.get(key)
        if slot is not None:
            return slot
    return slots_by_index.get(int(entry.generated_course_slot))


def _restore_entry(
    entry: TrackSamplingEntryConfig,
    slot: GeneratedXCupSlot,
) -> TrackSamplingEntryConfig:
    return entry.model_copy(
        update={
            "id": track_sampling_entry_id(
                course_id=slot.course_id,
                runtime_course_key=slot.course_key,
                mode=entry.mode,
                gp_difficulty=entry.gp_difficulty,
                vehicle=entry.vehicle,
            ),
            "runtime_course_key": slot.course_key,
            "course_id": slot.course_id,
            "course_name": slot.course_name,
            "display_name": slot.course_name,
            "baseline_state_path": None,
            "generated_course_hash": slot.course_hash,
            "generated_course_seed": slot.course_seed,
            "generated_course_generation": slot.generation,
            "generated_course_segment_count": slot.segment_count,
            "generated_course_length": slot.course_length,
            "log_per_course": False,
        }
    )


def _restore_entry_artifact(
    entry: TrackSamplingEntryConfig,
    artifact: TrackSamplingMaterializedArtifact,
) -> TrackSamplingEntryConfig:
    update: dict[str, object] = {
        "id": artifact.entry_id,
        "runtime_course_key": artifact.course_key,
        "baseline_state_path": artifact.baseline_state_path,
    }
    if artifact.source_course_index is not None:
        update["source_course_index"] = artifact.source_course_index
    if artifact.source_vehicle is not None:
        update["source_vehicle"] = artifact.source_vehicle
    if artifact.source_gp_difficulty is not None:
        update["source_gp_difficulty"] = artifact.source_gp_difficulty
    if artifact.source_engine_setting_raw_value is not None:
        update["source_engine_setting_raw_value"] = artifact.source_engine_setting_raw_value
    if artifact.generated_course_segment_count is not None:
        update["generated_course_segment_count"] = artifact.generated_course_segment_count
    if artifact.generated_course_length is not None:
        update["generated_course_length"] = artifact.generated_course_length
    if artifact.generated_course_slot is not None:
        update["generated_course_slot"] = artifact.generated_course_slot
    if artifact.generated_course_generation is not None:
        update["generated_course_generation"] = artifact.generated_course_generation
    if artifact.generated_course_id is not None:
        update["course_id"] = artifact.generated_course_id
    if artifact.generated_course_name is not None:
        update["course_name"] = artifact.generated_course_name
        update["display_name"] = artifact.generated_course_name
    if artifact.generated_course_hash is not None:
        update["generated_course_hash"] = artifact.generated_course_hash
    if artifact.generated_course_seed is not None:
        update["generated_course_seed"] = artifact.generated_course_seed
    return entry.model_copy(update=update)


def _artifact_matches_entry_identity(
    entry: TrackSamplingEntryConfig,
    artifact: TrackSamplingMaterializedArtifact,
) -> bool:
    if entry.generated_course_kind != X_CUP_COURSE.generated_kind:
        return True
    return (
        artifact.generated_course_slot == entry.generated_course_slot
        and artifact.generated_course_generation == entry.generated_course_generation
        and artifact.generated_course_id == entry.course_id
        and artifact.generated_course_hash == entry.generated_course_hash
        and artifact.generated_course_seed == entry.generated_course_seed
    )
