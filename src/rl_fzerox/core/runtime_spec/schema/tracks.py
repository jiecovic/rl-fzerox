# src/rl_fzerox/core/runtime_spec/schema/tracks.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, XCupGeneratedCourseKind
from rl_fzerox.core.runtime_spec.schema.common import TrackSamplingMode


class TrackRecordEntryConfig(BaseModel):
    """One human reference time for a track."""

    model_config = ConfigDict(extra="forbid")

    time_ms: PositiveInt
    player: str | None = None
    date: str | None = None
    mode: Literal["NTSC", "PAL"] | None = None

    def info(self, prefix: str) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {f"{prefix}_time_ms": int(self.time_ms)}
        if self.player is not None:
            info[f"{prefix}_player"] = self.player
        if self.date is not None:
            info[f"{prefix}_date"] = self.date
        if self.mode is not None:
            info[f"{prefix}_mode"] = self.mode
        return info


class TrackRecordsConfig(BaseModel):
    """External reference records for a track."""

    model_config = ConfigDict(extra="forbid")

    source_label: str = "F-Zero X WR History"
    source_url: str | None = None
    non_agg_best: TrackRecordEntryConfig | None = None
    non_agg_worst: TrackRecordEntryConfig | None = None

    def info(self) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {"track_record_source_label": self.source_label}
        if self.source_url is not None:
            info["track_record_source_url"] = self.source_url
        if self.non_agg_best is not None:
            info.update(self.non_agg_best.info("track_non_agg_best"))
        if self.non_agg_worst is not None:
            info.update(self.non_agg_worst.info("track_non_agg_worst"))
        return info


class TrackSamplingEntryConfig(BaseModel):
    """One reset-time baseline candidate for multi-track training."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    course_name: str | None = None
    baseline_state_path: Path | None = None
    weight: PositiveFloat = 1.0
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting: str | None = None
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting: str | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    generated_course_kind: XCupGeneratedCourseKind | None = None
    generated_course_seed: NonNegativeInt | None = None
    generated_course_hash: str | None = None
    generated_course_slot: NonNegativeInt | None = None
    generated_course_generation: NonNegativeInt | None = None
    generated_course_segment_count: NonNegativeInt | None = None
    generated_course_length: PositiveFloat | None = None
    log_per_course: bool = True
    records: TrackRecordsConfig | None = None

    @model_validator(mode="after")
    def _validate_generated_course_fields(self) -> TrackSamplingEntryConfig:
        if self.generated_course_kind is None:
            return self
        if self.generated_course_seed is None:
            raise ValueError("generated_course_seed is required for generated course entries")
        if not self.generated_course_hash:
            raise ValueError("generated_course_hash is required for generated course entries")
        if (
            self.generated_course_kind == X_CUP_COURSE.generated_kind
            and self.mode != X_CUP_COURSE.race_mode
        ):
            raise ValueError(f"generated X Cup entries must use mode={X_CUP_COURSE.race_mode}")
        if (
            self.generated_course_kind == X_CUP_COURSE.generated_kind
            and self.course_index != X_CUP_COURSE.course_index
        ):
            raise ValueError(
                f"generated X Cup entries must use course_index={X_CUP_COURSE.course_index}"
            )
        return self


class XCupRotationConfig(BaseModel):
    """Runtime policy for replacing solved generated X Cup slots."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    completion_threshold: float = Field(
        default=X_CUP_COURSE.rotation_defaults.completion_threshold,
        ge=0.0,
        le=1.0,
    )
    min_episodes: PositiveInt = X_CUP_COURSE.rotation_defaults.min_episodes
    min_completed_frames: PositiveInt = X_CUP_COURSE.rotation_defaults.min_completed_frames
    cooldown_episodes: NonNegativeInt = X_CUP_COURSE.rotation_defaults.cooldown_episodes


class TrackSamplingConfig(BaseModel):
    """Optional weighted baseline sampling performed at episode reset."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sampling_mode: TrackSamplingMode = "random"
    entries: tuple[TrackSamplingEntryConfig, ...] = ()
    step_balance_update_episodes: PositiveInt = 5
    step_balance_ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    step_balance_max_weight_scale: PositiveFloat = 5.0
    adaptive_step_balance_completion_weight: float = Field(default=0.35, ge=0.0)
    adaptive_step_balance_target_completion: float = Field(default=0.9, ge=0.0, le=1.0)
    adaptive_step_balance_min_confidence_episodes: PositiveInt = 24
    adaptive_step_balance_confidence_scale: PositiveFloat = 4.0
    step_balance_log_details: bool = False
    x_cup_rotation: XCupRotationConfig = Field(default_factory=XCupRotationConfig)

    @model_validator(mode="after")
    def _validate_entries_when_enabled(self) -> TrackSamplingConfig:
        if self.enabled and not self.entries:
            raise ValueError("env.track_sampling.entries must not be empty when enabled")
        if self.step_balance_max_weight_scale < 1.0:
            raise ValueError("track_sampling.step_balance_max_weight_scale must be >= 1.0")
        if self.adaptive_step_balance_confidence_scale < 1.0:
            raise ValueError("track_sampling.adaptive_step_balance_confidence_scale must be >= 1.0")
        return self


class TrackConfig(BaseModel):
    """Metadata for a concrete track/mode baseline."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    course_name: str | None = None
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting: str | None = None
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting: str | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    baseline_state_path: Path | None = None
    records: TrackRecordsConfig | None = None
    notes: str | None = None
