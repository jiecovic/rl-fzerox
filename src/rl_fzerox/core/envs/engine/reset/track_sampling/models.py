# src/rl_fzerox/core/envs/engine/reset/track_sampling/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.runtime_spec.schema import TrackRecordsConfig


@dataclass(frozen=True, slots=True)
class TrackSamplingLimits:
    """Limits that keep deterministic balanced sampling cycles bounded."""

    max_balanced_cycle_slots: int = 128


TRACK_SAMPLING_LIMITS = TrackSamplingLimits()


@dataclass(frozen=True, slots=True)
class TrackBaselineCacheLimits:
    """Per-worker limits for cached savestate bytes."""

    max_cached_state_bytes: int = 256 * 1024 * 1024


TRACK_BASELINE_CACHE_LIMITS = TrackBaselineCacheLimits()


@dataclass(frozen=True)
class SelectedTrack:
    """Reset-time track baseline selected for the current episode."""

    id: str
    display_name: str | None
    course_ref: str | None
    course_id: str | None
    runtime_course_key: str | None
    course_name: str | None
    baseline_state_path: Path
    weight: float
    course_index: int | None
    mode: str | None
    gp_difficulty: RaceDifficultyName | None
    vehicle: str | None
    vehicle_name: str | None
    engine_setting: str | None
    engine_setting_raw_value: int | None
    engine_setting_min_raw_value: int | None
    engine_setting_max_raw_value: int | None
    source_vehicle: str | None
    source_course_index: int | None
    source_gp_difficulty: RaceDifficultyName | None
    source_engine_setting: str | None
    source_engine_setting_raw_value: int | None
    generated_course_kind: str | None
    generated_course_seed: int | None
    generated_course_hash: str | None
    generated_course_slot: int | None
    generated_course_generation: int | None
    generated_course_segment_count: int | None
    generated_course_length: float | None
    log_per_course: bool
    records: TrackRecordsConfig | None
    sampling_mode: str
    cycle_position: int | None = None

    def info(self) -> dict[str, object]:
        info = {
            "track_sampling_enabled": True,
            "track_sampling_mode": self.sampling_mode,
            "track_id": self.id,
            "track_display_name": self.display_name,
            "track_course_ref": self.course_ref,
            "track_course_id": self.course_id,
            "track_runtime_course_key": self.runtime_course_key,
            "track_reset_course_key": self.runtime_course_key or self.course_id,
            "track_course_name": self.course_name,
            "track_baseline_state_path": str(self.baseline_state_path),
            "track_sampling_weight": self.weight,
            "track_course_index": self.course_index,
            "track_mode": self.mode,
            "track_gp_difficulty": self.gp_difficulty,
            "track_vehicle": self.vehicle,
            "track_vehicle_name": self.vehicle_name,
            "track_engine_setting": self.engine_setting,
            "track_engine_setting_raw_value": self.engine_setting_raw_value,
            "track_sampling_cycle_position": self.cycle_position,
            "track_generated_course_kind": self.generated_course_kind,
            "track_generated_course_seed": self.generated_course_seed,
            "track_generated_course_hash": self.generated_course_hash,
            "track_generated_course_slot": self.generated_course_slot,
            "track_generated_course_generation": self.generated_course_generation,
            "track_generated_course_segment_count": self.generated_course_segment_count,
            "track_generated_course_length": self.generated_course_length,
            "track_log_per_course": self.log_per_course,
        }
        if self.records is not None:
            info.update(self.records.info())
        return info
