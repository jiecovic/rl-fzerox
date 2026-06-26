# src/rl_fzerox/core/runtime_spec/schema/tracks/sampling.py
"""Reset-target sampling runtime schemas.

Track sampling turns the selected training pool into concrete reset entries and
per-reset scheduling parameters. Entries carry baseline, generated-course,
vehicle, engine, and logging metadata because the reset sampler, callbacks, and
engine tuner all need the same resolved target identity. The serialized entry
shape stays flat for manifest stability; typed metadata views group related
fields for code that needs to reason about source setup, variants, alternate
baselines, or generated courses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.domain.courses import X_CUP_COURSE, XCupGeneratedCourseKind
from rl_fzerox.core.domain.race import RaceDifficultyName
from rl_fzerox.core.runtime_spec.schema.common import (
    DeficitBudgetDifficultyMetric,
    TrackSamplingMode,
)
from rl_fzerox.core.runtime_spec.schema.tracks.engine_tuning import (
    AdaptiveEngineTuningConfig,
)
from rl_fzerox.core.runtime_spec.schema.tracks.records import TrackRecordsConfig
from rl_fzerox.core.runtime_spec.schema.tracks.x_cup import XCupRotationConfig
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_course_key,
    track_sampling_reset_target_key,
)


@dataclass(frozen=True, slots=True)
class TrackSamplingSourceSetupMetadata:
    """Source race setup used to regenerate or retarget a baseline state."""

    vehicle: str | None
    course_index: int | None
    gp_difficulty: RaceDifficultyName | None
    engine_setting_raw_value: int | None


@dataclass(frozen=True, slots=True)
class TrackSamplingBaselineVariantMetadata:
    """Opponent-grid variant metadata for materialized GP race baselines."""

    index: int | None
    count: int | None
    seed: int | None


@dataclass(frozen=True, slots=True)
class TrackSamplingAltBaselineMetadata:
    """Captured alternate baseline identity attached to a sampling entry."""

    id: str | None
    label: str | None
    source_entry_id: str | None


@dataclass(frozen=True, slots=True)
class TrackSamplingGeneratedCourseMetadata:
    """Generated-course identity and course-shape metadata for one entry."""

    kind: XCupGeneratedCourseKind | None
    seed: int | None
    course_hash: str | None
    slot: int | None
    generation: int | None
    segment_count: int | None
    course_length: float | None


class TrackSamplingEntryConfig(BaseModel):
    """One reset-time baseline candidate for multi-track training."""

    model_config = ConfigDict(extra="forbid")

    # Stable reset-target identity and human-facing course labels.
    id: str
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    runtime_course_key: str | None = None
    course_name: str | None = None
    # Materialized baseline and reset scheduler weight.
    baseline_state_path: Path | None = None
    weight: PositiveFloat = 1.0
    # Race setup resolved from registry/manager config.
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    # Source setup before generated-course or runtime restoration rewrites.
    source_vehicle: str | None = None
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    # Scheduler grouping for materialized baseline variants and alternate states.
    baseline_group_id: str | None = None
    baseline_group_weight: PositiveFloat | None = None
    baseline_variant_index: NonNegativeInt | None = None
    baseline_variant_count: PositiveInt | None = None
    baseline_variant_seed: NonNegativeInt | None = None
    alt_baseline_id: str | None = None
    alt_baseline_label: str | None = None
    alt_baseline_source_entry_id: str | None = None
    # Generated-course identity and rotation metadata, currently used by X-Cup.
    generated_course_kind: XCupGeneratedCourseKind | None = None
    generated_course_seed: NonNegativeInt | None = None
    generated_course_hash: str | None = None
    generated_course_slot: NonNegativeInt | None = None
    generated_course_generation: NonNegativeInt | None = None
    generated_course_segment_count: NonNegativeInt | None = None
    generated_course_length: PositiveFloat | None = None
    # Logging and display metadata consumed by training/watch surfaces.
    log_per_course: bool = True
    records: TrackRecordsConfig | None = None

    def source_setup_metadata(self) -> TrackSamplingSourceSetupMetadata:
        """Return source-race fields as one named value object."""

        return TrackSamplingSourceSetupMetadata(
            vehicle=self.source_vehicle,
            course_index=self.source_course_index,
            gp_difficulty=self.source_gp_difficulty,
            engine_setting_raw_value=self.source_engine_setting_raw_value,
        )

    def baseline_variant_metadata(self) -> TrackSamplingBaselineVariantMetadata | None:
        """Return baseline-variant metadata when any variant field is present."""

        if not _any_value_present(
            self.baseline_variant_index,
            self.baseline_variant_count,
            self.baseline_variant_seed,
        ):
            return None
        return TrackSamplingBaselineVariantMetadata(
            index=self.baseline_variant_index,
            count=self.baseline_variant_count,
            seed=self.baseline_variant_seed,
        )

    def alt_baseline_metadata(self) -> TrackSamplingAltBaselineMetadata | None:
        """Return alternate-baseline metadata when any alt field is present."""

        if not _any_value_present(
            self.alt_baseline_id,
            self.alt_baseline_label,
            self.alt_baseline_source_entry_id,
        ):
            return None
        return TrackSamplingAltBaselineMetadata(
            id=self.alt_baseline_id,
            label=self.alt_baseline_label,
            source_entry_id=self.alt_baseline_source_entry_id,
        )

    def generated_course_metadata(self) -> TrackSamplingGeneratedCourseMetadata | None:
        """Return generated-course metadata without changing the serialized shape."""

        if not _any_value_present(
            self.generated_course_kind,
            self.generated_course_seed,
            self.generated_course_hash,
            self.generated_course_slot,
            self.generated_course_generation,
            self.generated_course_segment_count,
            self.generated_course_length,
        ):
            return None
        return TrackSamplingGeneratedCourseMetadata(
            kind=self.generated_course_kind,
            seed=self.generated_course_seed,
            course_hash=self.generated_course_hash,
            slot=self.generated_course_slot,
            generation=self.generated_course_generation,
            segment_count=self.generated_course_segment_count,
            course_length=self.generated_course_length,
        )

    def course_key(self) -> str:
        """Return the stable course/slot key used for stats and watch navigation."""

        return track_sampling_course_key(
            entry_id=self.id,
            course_id=self.course_id,
            runtime_course_key=self.runtime_course_key,
            course_ref=self.course_ref,
            course_index=self.course_index,
        )

    def reset_target_key(self) -> str:
        """Return the reset-target key, including GP difficulty when applicable."""

        return track_sampling_reset_target_key(
            entry_id=self.id,
            course_id=self.course_id,
            runtime_course_key=self.runtime_course_key,
            course_ref=self.course_ref,
            course_index=self.course_index,
            gp_difficulty=self.gp_difficulty,
        )

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


def _any_value_present(*values: object | None) -> bool:
    return any(value is not None for value in values)


class TrackSamplingConfig(BaseModel):
    """Optional weighted baseline sampling performed at episode reset."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sampling_mode: TrackSamplingMode = "equal"
    entries: tuple[TrackSamplingEntryConfig, ...] = ()
    step_balance_update_episodes: PositiveInt = 5
    step_balance_ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    step_balance_max_weight_scale: PositiveFloat = 5.0
    deficit_budget_uniform_fraction: float = Field(default=0.7, ge=0.0, le=1.0)
    deficit_budget_focus_sharpness: float = Field(default=1.0, ge=0.0)
    deficit_budget_ema_alpha: float = Field(default=0.02, gt=0.0, le=1.0)
    deficit_budget_weight_update_rollouts: PositiveInt = 20
    deficit_budget_difficulty_metric: DeficitBudgetDifficultyMetric = "completion_ema"
    deficit_budget_warmup_min_episodes_per_course: int = Field(default=10, ge=0)
    deficit_budget_uniform_staleness_rotations: float = Field(default=2.0, ge=0.0)
    baseline_variant_count: PositiveInt = Field(default=1, le=16)
    step_balance_log_details: bool = False
    x_cup_rotation: XCupRotationConfig = Field(default_factory=XCupRotationConfig)
    engine_tuning: AdaptiveEngineTuningConfig = Field(default_factory=AdaptiveEngineTuningConfig)

    @model_validator(mode="after")
    def _validate_entries_when_enabled(self) -> TrackSamplingConfig:
        if self.enabled and not self.entries:
            raise ValueError("env.track_sampling.entries must not be empty when enabled")
        if self.step_balance_max_weight_scale < 1.0:
            raise ValueError("track_sampling.step_balance_max_weight_scale must be >= 1.0")
        return self
