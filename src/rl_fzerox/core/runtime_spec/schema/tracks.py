# src/rl_fzerox/core/runtime_spec/schema/tracks.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from rl_fzerox.core.domain.engine_setting import ENGINE_SLIDER_STEP_MAX
from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, XCupGeneratedCourseKind
from rl_fzerox.core.engine_tuning.types import (
    ENGINE_TUNER_DEFAULTS,
    EngineTunerBackend,
    EngineTunerObjective,
    engine_bucket_candidates,
)
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
    runtime_course_key: str | None = None
    course_name: str | None = None
    baseline_state_path: Path | None = None
    weight: PositiveFloat = 1.0
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    gp_difficulty: RaceDifficultyName | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    source_vehicle: str | None = None
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    baseline_group_id: str | None = None
    baseline_group_weight: PositiveFloat | None = None
    alt_baseline_id: str | None = None
    alt_baseline_label: str | None = None
    alt_baseline_source_entry_id: str | None = None
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
    max_episodes: PositiveInt | None = X_CUP_COURSE.rotation_defaults.max_episodes
    ema_alpha: PositiveFloat = Field(
        default=X_CUP_COURSE.rotation_defaults.ema_alpha,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_episode_window(self) -> XCupRotationConfig:
        if self.max_episodes is not None and self.max_episodes < self.min_episodes:
            raise ValueError("x_cup_rotation.max_episodes must be >= min_episodes")
        return self


class AdaptiveEngineTuningConfig(BaseModel):
    """Reset-time adaptive engine-setting sampler configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_raw_value: NonNegativeInt = Field(default=0, le=ENGINE_SLIDER_STEP_MAX)
    max_raw_value: NonNegativeInt = Field(
        default=ENGINE_SLIDER_STEP_MAX,
        le=ENGINE_SLIDER_STEP_MAX,
    )
    backend: EngineTunerBackend = ENGINE_TUNER_DEFAULTS.backend
    objective: EngineTunerObjective = ENGINE_TUNER_DEFAULTS.objective
    reward_fingerprint: str | None = None
    slider_spacing: PositiveInt = Field(
        default=ENGINE_TUNER_DEFAULTS.bandit_slider_spacing,
        le=ENGINE_SLIDER_STEP_MAX,
    )
    stat_decay: float = Field(default=ENGINE_TUNER_DEFAULTS.stat_decay, gt=0.0, lt=1.0)
    prior_finish_time_seconds: PositiveFloat = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    exploration_scale: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.exploration_seconds
    ensemble_members: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_ensemble_members
    randomized_prior_seconds: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.mlp_randomized_prior_seconds
    hidden_dim: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_hidden_dim
    training_steps: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_training_steps
    learning_rate: PositiveFloat = ENGINE_TUNER_DEFAULTS.mlp_learning_rate
    bootstrap_keep_probability: float = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_bootstrap_keep_probability,
        gt=0.0,
        le=1.0,
    )
    warmup_successes: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_warmup_successes
    observation_noise_seconds: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: PositiveFloat = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw
    uniform_exploration: float = Field(
        default=ENGINE_TUNER_DEFAULTS.uniform_exploration,
        ge=0.0,
        le=1.0,
    )
    greedy_plateau_tolerance_seconds: NonNegativeFloat = (
        ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds
    )

    @model_serializer(mode="wrap")
    def _serialize_engine_tuning(self, handler: SerializerFunctionWrapHandler) -> object:
        data = handler(self)
        if isinstance(data, dict) and self.backend != "bandit":
            data.pop("objective", None)
            data.pop("reward_fingerprint", None)
            data.pop("slider_spacing", None)
        if isinstance(data, dict) and self.backend == "bandit":
            data.pop("greedy_plateau_tolerance_seconds", None)
        if isinstance(data, dict) and self.backend != "gaussian_process":
            data.pop("stat_decay", None)
            data.pop("observation_noise_seconds", None)
            data.pop("curve_lengthscale_raw", None)
        if isinstance(data, dict) and self.backend not in {"bandit", "gaussian_process"}:
            data.pop("exploration_scale", None)
        if isinstance(data, dict) and self.backend != "mlp_ensemble":
            data.pop("ensemble_members", None)
            data.pop("randomized_prior_seconds", None)
            data.pop("hidden_dim", None)
            data.pop("training_steps", None)
            data.pop("learning_rate", None)
            data.pop("bootstrap_keep_probability", None)
            data.pop("warmup_successes", None)
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_bandit_spacing(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        next_data = dict(data)
        old_spacing = next_data.pop("bucket_size", None)
        if old_spacing is not None:
            next_data.setdefault("slider_spacing", old_spacing)
        return next_data

    @model_validator(mode="after")
    def _validate_engine_range(self) -> AdaptiveEngineTuningConfig:
        if self.min_raw_value > self.max_raw_value:
            raise ValueError("engine_tuning.min_raw_value must be <= max_raw_value")
        if self.enabled and self.backend == "bandit":
            engine_bucket_candidates(
                minimum=self.min_raw_value,
                maximum=self.max_raw_value,
                slider_spacing=self.slider_spacing,
            )
        return self


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
    deficit_budget_uniform_fraction: float = Field(default=0.7, ge=0.0, le=1.0)
    deficit_budget_focus_sharpness: float = Field(default=1.0, ge=0.0)
    deficit_budget_ema_alpha: float = Field(default=0.02, gt=0.0, le=1.0)
    deficit_budget_weight_update_rollouts: PositiveInt = 20
    step_balance_log_details: bool = False
    x_cup_rotation: XCupRotationConfig = Field(default_factory=XCupRotationConfig)
    engine_tuning: AdaptiveEngineTuningConfig = Field(default_factory=AdaptiveEngineTuningConfig)

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
    engine_setting_raw_value: NonNegativeInt | None = None
    engine_setting_min_raw_value: NonNegativeInt | None = None
    engine_setting_max_raw_value: NonNegativeInt | None = None
    source_course_index: NonNegativeInt | None = None
    source_gp_difficulty: RaceDifficultyName | None = None
    source_engine_setting_raw_value: NonNegativeInt | None = None
    baseline_state_path: Path | None = None
    records: TrackRecordsConfig | None = None
    notes: str | None = None
