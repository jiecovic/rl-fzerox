# src/rl_fzerox/core/manager/run_spec/sections/environment.py
"""Environment/runtime section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt, model_validator

from rl_fzerox.core.domain.race import CameraSettingName
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, RendererName


class ManagedEnvironmentConfig(BaseModel):
    """Episode-limit and emulator-runtime knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    max_episode_steps: PositiveInt = 12_000
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    renderer: RendererName = DEFAULT_RENDERER
    camera_setting: CameraSettingName = "close_behind"
    randomize_gp_lives_on_reset: bool = True
    gp_lives_jitter_min: int = -2
    gp_lives_jitter_max: int = 3

    @model_validator(mode="after")
    def _validate_gp_lives_jitter_bounds(self) -> ManagedEnvironmentConfig:
        if self.gp_lives_jitter_min > self.gp_lives_jitter_max:
            raise ValueError("gp_lives_jitter_min must be <= gp_lives_jitter_max")
        return self
