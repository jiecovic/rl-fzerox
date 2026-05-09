# src/rl_fzerox/core/manager/run_spec/sections/action.py
"""Action-space section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.manager.run_spec.common import (
    ActionAxisMode,
    ActionDriveMode,
    LeanOutputMode,
)


class ManagedActionConfig(BaseModel):
    """Action-space knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 2
    steering_mode: ActionAxisMode = "continuous"
    steer_buckets: int = Field(default=7, ge=3)
    drive_mode: ActionDriveMode = "on_off"
    force_full_throttle: bool = False
    continuous_drive_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_drive_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_drive_min_thrust: float = Field(default=0.25, ge=0.0, le=1.0)
    include_air_brake: bool = True
    air_brake_mode: ActionDriveMode = "on_off"
    enable_air_brake: bool = True
    mask_air_brake_on_ground: bool = False
    continuous_air_brake_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_air_brake_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_air_brake_min_duty: float = Field(default=0.0, ge=0.0, le=1.0)
    include_boost: bool = True
    enable_boost: bool = True
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    include_lean: bool = True
    enable_lean: bool = True
    lean_output_mode: LeanOutputMode = "three_way"
    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    lean_initial_lockout_frames: NonNegativeInt = 0
    include_pitch: bool = True
    enable_pitch: bool = True
    pitch_mode: ActionAxisMode = "discrete"
    pitch_deadzone: float = Field(default=0.1, ge=0.0, lt=1.0)
    pitch_buckets: int = Field(default=5, ge=3)

    @field_validator("steer_buckets", "pitch_buckets")
    @classmethod
    def _validate_odd_bucket_count(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("action bucket counts must be odd")
        return value

    @model_validator(mode="after")
    def _validate_supported_layout(self) -> ManagedActionConfig:
        if self.continuous_drive_deadzone >= self.continuous_drive_full_threshold:
            raise ValueError(
                "continuous_drive_deadzone must be lower than continuous_drive_full_threshold"
            )
        if self.continuous_air_brake_deadzone >= self.continuous_air_brake_full_threshold:
            raise ValueError(
                "continuous_air_brake_deadzone must be lower than "
                "continuous_air_brake_full_threshold"
            )
        if not self.include_air_brake:
            self.enable_air_brake = False
        if not self.include_boost:
            self.enable_boost = False
        if not self.include_lean:
            self.enable_lean = False
        elif self.lean_output_mode == "independent_buttons":
            self.lean_mode = "raw"
        if not self.include_pitch:
            self.enable_pitch = False
        elif self.pitch_mode == "continuous":
            self.enable_pitch = True
        return self
