# src/rl_fzerox/core/training/session/callbacks/track_sampling/artifacts.py
"""Materialized baseline artifact snapshots derived from track-sampling config."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig


@dataclass(frozen=True, slots=True)
class TrackSamplingMaterializedArtifact:
    """One active reset artifact for a materialized track-sampling entry."""

    course_key: str
    reset_variant_key: str
    entry_id: str
    baseline_state_path: Path
    metadata_path: Path
    source_course_index: int | None
    source_gp_difficulty: str | None
    source_vehicle: str | None
    source_engine_setting_raw_value: int | None
    generated_course_slot: int | None
    generated_course_generation: int | None
    generated_course_id: str | None
    generated_course_name: str | None
    generated_course_hash: str | None
    generated_course_seed: int | None
    generated_course_segment_count: int | None
    generated_course_length: float | None


def materialized_track_sampling_artifacts(
    config: TrackSamplingConfig,
) -> tuple[TrackSamplingMaterializedArtifact, ...]:
    """Return active reset artifacts from one materialized track-sampling config."""

    return tuple(
        artifact
        for entry in config.entries
        if (artifact := materialized_track_sampling_artifact(entry)) is not None
    )


def materialized_track_sampling_artifact(
    entry: TrackSamplingEntryConfig,
) -> TrackSamplingMaterializedArtifact | None:
    """Project one materialized track-sampling entry into artifact state."""

    if entry.baseline_state_path is None:
        return None
    baseline_state_path = entry.baseline_state_path.expanduser().resolve()
    return TrackSamplingMaterializedArtifact(
        course_key=track_sampling_artifact_course_key(entry),
        reset_variant_key=track_sampling_artifact_reset_variant_key(entry),
        entry_id=entry.id,
        baseline_state_path=baseline_state_path,
        metadata_path=baseline_state_path.with_suffix(".json"),
        source_course_index=entry.source_course_index,
        source_gp_difficulty=entry.source_gp_difficulty or entry.gp_difficulty,
        source_vehicle=entry.source_vehicle or entry.vehicle,
        source_engine_setting_raw_value=entry.source_engine_setting_raw_value,
        generated_course_slot=entry.generated_course_slot,
        generated_course_generation=entry.generated_course_generation,
        generated_course_id=entry.course_id,
        generated_course_name=entry.course_name or entry.display_name,
        generated_course_hash=entry.generated_course_hash,
        generated_course_seed=entry.generated_course_seed,
        generated_course_segment_count=entry.generated_course_segment_count,
        generated_course_length=entry.generated_course_length,
    )


def track_sampling_artifact_index(
    artifacts: Iterable[TrackSamplingMaterializedArtifact],
) -> dict[tuple[str, str], TrackSamplingMaterializedArtifact]:
    return {(artifact.course_key, artifact.reset_variant_key): artifact for artifact in artifacts}


def track_sampling_artifact_course_key(entry: TrackSamplingEntryConfig) -> str:
    return entry.runtime_course_key or entry.course_id or entry.id


def track_sampling_artifact_reset_variant_key(entry: TrackSamplingEntryConfig) -> str:
    return reset_variant_key(
        mode=entry.mode,
        gp_difficulty=entry.gp_difficulty or entry.source_gp_difficulty,
        vehicle=entry.vehicle or entry.source_vehicle,
    )


def reset_variant_key(
    *,
    mode: str | None,
    gp_difficulty: str | None,
    vehicle: str | None,
) -> str:
    return "|".join(
        (
            _variant_part("mode", mode),
            _variant_part("gp_difficulty", gp_difficulty),
            _variant_part("vehicle", vehicle),
        )
    )


def _variant_part(name: str, value: str | None) -> str:
    return f"{name}={'' if value is None else value}"
