# src/rl_fzerox/core/envs/engine/reset/track_sampling/models.py
"""Internal data models for sampled track reset selection.

`SelectedTrack` carries the reset baseline plus the metadata emitted into env
info, training stats, and engine-tuning contexts for one episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.domain.race import RaceDifficultyName
from rl_fzerox.core.runtime_spec.schema import TrackRecordsConfig
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_course_key,
    track_sampling_reset_target_key,
)


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


TrackSamplingDeficitLane = Literal["uniform", "adaptive"]


@dataclass(frozen=True, slots=True)
class TrackSamplingQueuedReset:
    """One externally scheduled reset slot for deficit-budget sampling."""

    course_id: str
    deficit_lane: TrackSamplingDeficitLane | None = None


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
    engine_setting_raw_value: int | None
    engine_setting_min_raw_value: int | None
    engine_setting_max_raw_value: int | None
    engine_tuning_context_key: str | None
    engine_tuning_course_key: str | None
    engine_tuning_vehicle_id: str | None
    engine_tuning_sampled_score: float | None
    engine_tuning_mean_score: float | None
    engine_tuning_uncertainty_score: float | None
    engine_tuning_finish_count: int | None
    source_vehicle: str | None
    source_course_index: int | None
    source_gp_difficulty: RaceDifficultyName | None
    source_engine_setting_raw_value: int | None
    baseline_group_id: str | None
    baseline_group_weight: float | None
    baseline_variant_index: int | None
    baseline_variant_count: int | None
    baseline_variant_seed: int | None
    alt_baseline_id: str | None
    alt_baseline_label: str | None
    alt_baseline_source_entry_id: str | None
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
    deficit_budget_lane: TrackSamplingDeficitLane | None = None

    def info(self) -> dict[str, object]:
        source_entry_id = self._source_entry_id()
        course_key = self._course_key(source_entry_id)
        info = self._identity_info(source_entry_id, course_key)
        info.update(self._engine_tuning_info())
        info.update(self._sampling_state_info())
        info.update(self._baseline_info())
        info.update(self._generated_course_info())
        info["track_log_per_course"] = self.log_per_course
        if self.records is not None:
            info.update(self.records.info())
        return info

    def _source_entry_id(self) -> str:
        return self.alt_baseline_source_entry_id or self.id

    def _course_key(self, source_entry_id: str) -> str:
        return track_sampling_course_key(
            entry_id=source_entry_id,
            course_id=self.course_id,
            runtime_course_key=self.runtime_course_key,
            course_ref=self.course_ref,
            course_index=self.course_index,
        )

    def _identity_info(
        self,
        source_entry_id: str,
        course_key: str,
    ) -> dict[str, object]:
        return {
            "track_sampling_enabled": True,
            "track_sampling_mode": self.sampling_mode,
            "track_entry_id": self.id,
            "track_id": source_entry_id,
            "track_course_key": course_key,
            "track_display_name": self.display_name,
            "track_course_ref": self.course_ref,
            "track_course_id": self.course_id,
            "track_runtime_course_key": self.runtime_course_key,
            "track_reset_course_key": self.runtime_course_key or self.course_id,
            "track_reset_target_key": track_sampling_reset_target_key(
                entry_id=source_entry_id,
                course_id=self.course_id,
                runtime_course_key=self.runtime_course_key,
                course_ref=self.course_ref,
                course_index=self.course_index,
                gp_difficulty=self.gp_difficulty,
            ),
            "track_course_name": self.course_name,
            "track_baseline_state_path": str(self.baseline_state_path),
            "track_sampling_weight": self.weight,
            "track_course_index": self.course_index,
            "track_mode": self.mode,
            "track_gp_difficulty": self.gp_difficulty,
            "track_vehicle": self.vehicle,
            "track_vehicle_name": self.vehicle_name,
            "track_engine_setting_raw_value": self.engine_setting_raw_value,
        }

    def _engine_tuning_info(self) -> dict[str, object]:
        return {
            "engine_tuning_context_key": self.engine_tuning_context_key,
            "engine_tuning_course_key": self.engine_tuning_course_key,
            "engine_tuning_vehicle_id": self.engine_tuning_vehicle_id,
            "engine_tuning_sampled_score": self.engine_tuning_sampled_score,
            "engine_tuning_mean_score": self.engine_tuning_mean_score,
            "engine_tuning_uncertainty_score": self.engine_tuning_uncertainty_score,
            "engine_tuning_finish_count": self.engine_tuning_finish_count,
        }

    def _sampling_state_info(self) -> dict[str, object]:
        return {
            "track_sampling_cycle_position": self.cycle_position,
            "track_sampling_deficit_lane": self.deficit_budget_lane,
        }

    def _baseline_info(self) -> dict[str, object]:
        return {
            "track_baseline_group_id": self.baseline_group_id,
            "track_baseline_group_weight": self.baseline_group_weight,
            "track_baseline_variant_index": self.baseline_variant_index,
            "track_baseline_variant_count": self.baseline_variant_count,
            "track_baseline_variant_seed": self.baseline_variant_seed,
            "track_alt_baseline_id": self.alt_baseline_id,
            "track_alt_baseline_label": self.alt_baseline_label,
            "track_alt_baseline_source_entry_id": self.alt_baseline_source_entry_id,
        }

    def _generated_course_info(self) -> dict[str, object]:
        return {
            "track_generated_course_kind": self.generated_course_kind,
            "track_generated_course_seed": self.generated_course_seed,
            "track_generated_course_hash": self.generated_course_hash,
            "track_generated_course_slot": self.generated_course_slot,
            "track_generated_course_generation": self.generated_course_generation,
            "track_generated_course_segment_count": self.generated_course_segment_count,
            "track_generated_course_length": self.generated_course_length,
        }
