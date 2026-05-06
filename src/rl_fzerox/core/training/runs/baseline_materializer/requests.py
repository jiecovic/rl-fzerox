# src/rl_fzerox/core/training/runs/baseline_materializer/requests.py
from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import TrackConfig, TrackSamplingEntryConfig

from .models import BaselineRequest


def can_generate_track_config(track: TrackConfig) -> bool:
    return (
        track.course_index is not None
        and track.mode is not None
        and track.vehicle is not None
        and track.engine_setting_raw_value is not None
    )


def request_from_track_config(
    track: TrackConfig,
    *,
    camera_setting: str | None,
    fallback_label: str,
) -> BaselineRequest:
    label = _baseline_request_label(
        mode=track.mode,
        course_id=track.course_id or track.id,
        course_name=track.course_name or track.display_name,
        course_index=track.course_index,
        vehicle_id=track.vehicle,
        vehicle_name=track.vehicle_name,
        fallback_label=fallback_label,
    )
    return BaselineRequest(
        label=label,
        source_state_path=track.baseline_state_path,
        course_id=track.course_id or track.id,
        course_name=track.course_name or track.display_name,
        course_index=track.course_index,
        mode=track.mode,
        vehicle=track.vehicle,
        vehicle_name=track.vehicle_name,
        source_vehicle=track.source_vehicle,
        engine_setting=track.engine_setting,
        engine_setting_raw_value=track.engine_setting_raw_value,
        source_course_index=track.source_course_index,
        source_engine_setting=track.source_engine_setting,
        source_engine_setting_raw_value=track.source_engine_setting_raw_value,
        camera_setting=camera_setting,
    )


def request_from_track_entry(
    entry: TrackSamplingEntryConfig,
    *,
    camera_setting: str | None,
) -> BaselineRequest:
    label = _baseline_request_label(
        mode=entry.mode,
        course_id=entry.course_id,
        course_name=entry.course_name or entry.display_name,
        course_index=entry.course_index,
        vehicle_id=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        fallback_label=entry.id,
    )
    return BaselineRequest(
        label=label,
        source_state_path=entry.baseline_state_path,
        course_id=entry.course_id,
        course_name=entry.course_name or entry.display_name,
        course_index=entry.course_index,
        mode=entry.mode,
        vehicle=entry.vehicle,
        vehicle_name=entry.vehicle_name,
        source_vehicle=entry.source_vehicle,
        engine_setting=entry.engine_setting,
        engine_setting_raw_value=entry.engine_setting_raw_value,
        source_course_index=entry.source_course_index,
        source_engine_setting=entry.source_engine_setting,
        source_engine_setting_raw_value=entry.source_engine_setting_raw_value,
        camera_setting=camera_setting,
    )


def _baseline_request_label(
    *,
    mode: str | None,
    course_id: str | None,
    course_name: str | None,
    course_index: int | None,
    vehicle_id: str | None,
    vehicle_name: str | None,
    fallback_label: str,
) -> str:
    course_part = (
        course_id or course_name or (None if course_index is None else f"course_{course_index:02d}")
    )
    vehicle_part = vehicle_id or vehicle_name
    if course_part is None or vehicle_part is None or mode is None:
        return fallback_label
    return f"{course_part}_{mode}_{vehicle_part}"
