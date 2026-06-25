# src/rl_fzerox/core/runtime_spec/schema/env.py
"""Runtime environment rollout configuration model.

This module owns env step/reset controls plus the nested action, observation,
and track-sampling sections used to construct F-Zero X Gymnasium environments.
Reward shaping and emulator boot paths live in sibling schema modules.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.domain.race import CameraSettingName
from rl_fzerox.core.runtime_spec.schema.actions import ActionConfig
from rl_fzerox.core.runtime_spec.schema.observations import ObservationConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingConfig


class EnvConfig(BaseModel):
    """Environment-level rollout settings that affect frame stepping."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 3
    # The step-like env limits below are counted per internal telemetry sample,
    # i.e. once per emulated frame, not once per outer env.step().
    max_episode_steps: PositiveInt = 12_000
    stuck_min_speed_kph: NonNegativeFloat = 50.0
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    terminate_on_energy_depleted: bool = False
    randomize_game_rng_on_reset: bool = False
    randomize_game_rng_requires_race_mode: bool = True
    randomize_gp_lives_on_reset: bool = True
    gp_lives_jitter_min: int = -2
    gp_lives_jitter_max: int = 3
    camera_setting: CameraSettingName | None = None
    reset_to_race: bool = False
    race_intro_target_timer: int | None = Field(default=39, ge=0, le=460)
    cache_track_baselines: bool = True
    track_sampling: TrackSamplingConfig = Field(default_factory=TrackSamplingConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)

    @model_validator(mode="after")
    def _validate_action_aware_observation_features(self) -> EnvConfig:
        from rl_fzerox.core.envs.observations.state.components import state_component_features

        if self.observation.state_components is None:
            return self
        for component in self.observation.state_components:
            state_component_features(
                component.data(),
                split_lean_history=self.action.runtime().split_lean_history,
            )
        return self

    @model_validator(mode="after")
    def _validate_gp_lives_jitter_bounds(self) -> EnvConfig:
        if self.gp_lives_jitter_min > self.gp_lives_jitter_max:
            raise ValueError("gp_lives_jitter_min must be <= gp_lives_jitter_max")
        return self
