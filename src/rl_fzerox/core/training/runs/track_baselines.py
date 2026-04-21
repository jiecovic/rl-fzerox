# src/rl_fzerox/core/training/runs/track_baselines.py
from __future__ import annotations

from rl_fzerox.core.config.schema import (
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


def single_active_track_entry(
    config: TrackSamplingConfig,
) -> TrackSamplingEntryConfig | None:
    """Return the only active sampled baseline, if the run is effectively single-track."""

    if not config.enabled or len(config.entries) != 1:
        return None
    return config.entries[0]


def track_config_from_sampling_entry(
    entry: TrackSamplingEntryConfig,
    *,
    fallback: TrackConfig,
) -> TrackConfig:
    """Project one materialized sampling entry into canonical top-level track metadata."""

    return fallback.model_copy(
        update={
            "id": entry.id,
            "display_name": entry.display_name,
            "course_ref": entry.course_ref,
            "course_id": entry.course_id,
            "course_name": entry.course_name,
            "course_index": entry.course_index,
            "mode": entry.mode,
            "vehicle": entry.vehicle,
            "vehicle_name": entry.vehicle_name,
            "source_vehicle": entry.source_vehicle,
            "engine_setting": entry.engine_setting,
            "engine_setting_raw_value": entry.engine_setting_raw_value,
            "source_course_index": entry.source_course_index,
            "source_engine_setting": entry.source_engine_setting,
            "source_engine_setting_raw_value": entry.source_engine_setting_raw_value,
            "baseline_state_path": entry.baseline_state_path,
            "records": entry.records,
        }
    )
