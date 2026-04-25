# src/rl_fzerox/core/config/schema_models/curriculum.py
from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.config.schema_models.actions import ActionMaskConfig
from rl_fzerox.core.config.schema_models.tracks import TrackSamplingConfig


class CurriculumTrainOverridesConfig(BaseModel):
    """Training hyperparameter overrides applied while one stage is active."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: PositiveFloat | None = None
    n_epochs: PositiveInt | None = None
    batch_size: PositiveInt | None = None
    clip_range: PositiveFloat | None = None
    ent_coef: NonNegativeFloat | None = None


class PerTrackLapsCompletedTriggerConfig(BaseModel):
    """Promotion gate requiring broad per-track lap progress."""

    model_config = ConfigDict(extra="forbid")

    mean_gte: NonNegativeFloat
    min_track_fraction_gte: float = Field(default=0.75, gt=0.0, le=1.0)
    min_episodes_per_track: PositiveInt = 1


class CurriculumTriggerConfig(BaseModel):
    """Episode-smoothed promotion condition for one curriculum stage."""

    model_config = ConfigDict(extra="forbid")

    race_laps_completed_mean_gte: NonNegativeFloat | None = None
    per_track_laps_completed: PerTrackLapsCompletedTriggerConfig | None = None

    @model_validator(mode="after")
    def _validate_has_trigger(self) -> CurriculumTriggerConfig:
        if self.race_laps_completed_mean_gte is None and self.per_track_laps_completed is None:
            raise ValueError(
                "Curriculum stage triggers must set race_laps_completed_mean_gte "
                "or per_track_laps_completed"
            )
        return self


class CurriculumStageConfig(BaseModel):
    """One curriculum stage with optional promotion trigger and overrides."""

    model_config = ConfigDict(extra="forbid")

    name: str
    until: CurriculumTriggerConfig | None = None
    action_mask: ActionMaskConfig | None = None
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    boost_min_energy_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    track_sampling: TrackSamplingConfig | None = None
    train: CurriculumTrainOverridesConfig | None = None


class CurriculumConfig(BaseModel):
    """Optional stage-based action-mask curriculum for training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    smoothing_episodes: PositiveInt = 1
    min_stage_episodes: PositiveInt = 1
    stages: tuple[CurriculumStageConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_enabled_stages(self) -> CurriculumConfig:
        if self.enabled and len(self.stages) == 0:
            raise ValueError("Enabled curriculum requires at least one stage")
        return self
