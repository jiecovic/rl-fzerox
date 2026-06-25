# src/rl_fzerox/core/runtime_spec/schema/tracks/x_cup.py
"""Generated X-Cup rotation runtime schema.

Rotation settings decide when generated X-Cup slots can be replaced during
training. The generated-course identity carried by individual sampling entries
lives in ``sampling.py`` so reset selection can validate the full entry shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from rl_fzerox.core.domain.courses import X_CUP_COURSE


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
