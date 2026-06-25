# src/rl_fzerox/core/training/runs/baseline_materializer/projection.py
"""Projection helpers that write materialized baseline metadata onto configs."""

from __future__ import annotations

from rl_fzerox.core.training.runs.baseline_materializer.models import (
    BaselineArtifact,
    BaselineRequest,
)


def baseline_artifact_entry_update(
    *,
    artifact: BaselineArtifact,
    request: BaselineRequest | None = None,
) -> dict[str, object]:
    """Project materialized baseline metadata onto track config update fields."""

    update: dict[str, object] = {"baseline_state_path": artifact.state_path}
    source_course_index = artifact.source_course_index
    if source_course_index is None and request is not None:
        source_course_index = request.course_index
    if source_course_index is not None:
        update["source_course_index"] = source_course_index

    source_vehicle = artifact.source_vehicle
    if source_vehicle is None and request is not None:
        source_vehicle = request.vehicle
    if source_vehicle is not None:
        update["source_vehicle"] = source_vehicle

    source_gp_difficulty = artifact.source_gp_difficulty
    if source_gp_difficulty is None and request is not None:
        source_gp_difficulty = request.gp_difficulty
    if source_gp_difficulty is not None:
        update["source_gp_difficulty"] = source_gp_difficulty

    source_engine_setting_raw_value = artifact.source_engine_setting_raw_value
    if source_engine_setting_raw_value is None and request is not None:
        source_engine_setting_raw_value = request.engine_setting_raw_value
    if source_engine_setting_raw_value is not None:
        update["source_engine_setting_raw_value"] = source_engine_setting_raw_value

    if artifact.generated_course_segment_count is not None:
        update["generated_course_segment_count"] = artifact.generated_course_segment_count
    if artifact.generated_course_length is not None:
        update["generated_course_length"] = artifact.generated_course_length
    return update
