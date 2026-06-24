# src/rl_fzerox/core/runtime_spec/schema/apps.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS, TrainAlgorithmName
from rl_fzerox.core.runtime_spec.schema.common import WatchFpsSetting
from rl_fzerox.core.runtime_spec.schema.env import EmulatorConfig, EnvConfig, RewardConfig
from rl_fzerox.core.runtime_spec.schema.policy import PolicyConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackConfig
from rl_fzerox.core.runtime_spec.schema.training import TrainConfig


class CareerModeRaceSetupConfig(BaseModel):
    """Resolved race setup used by the Career Mode menu runner."""

    model_config = ConfigDict(extra="forbid")

    difficulty: str
    cup_id: str
    course_id: str | None = None
    vehicle_id: str
    vehicle_display_name: str
    character_index: NonNegativeInt
    machine_select_slot: NonNegativeInt
    machine_select_row: NonNegativeInt
    machine_select_column: NonNegativeInt
    engine_setting_raw_value: int


class WatchRecordingConfig(BaseModel):
    """Optional recording output for watch or Career Mode playback."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: Path | None = None
    session_mp4_enabled: bool = True
    keep_failed_segments: bool = True
    render_input_hud: bool = False
    upscale_factor: int = Field(default=1, ge=1, le=4)


class WatchCareerDebugConfig(BaseModel):
    """Optional Career Mode controller trace output."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    screenshots: bool = True
    screenshot_limit: NonNegativeInt = 256


class WatchConfig(BaseModel):
    """Human-facing watch UI settings."""

    model_config = ConfigDict(extra="forbid")

    episodes: int | None = Field(default=None, gt=0)
    control_fps: WatchFpsSetting | None = None
    render_fps: WatchFpsSetting | None = None
    deterministic_policy: bool = True
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    attempt_seed: int | None = Field(default=None, ge=0, le=(1 << 32) - 1)
    policy_run_dir: Path | None = None
    policy_artifact: Literal["latest", "best", "final"] = "latest"
    policy_algorithm: TrainAlgorithmName | None = None
    lineage_frame_offset: NonNegativeInt | None = None
    manager_db_path: Path | None = None
    managed_run_id: str | None = None
    managed_save_game_id: str | None = None
    save_attempt_id: str | None = None
    single_save_target: bool = False
    single_save_target_perfect: bool = False
    single_save_target_clear_goal: NonNegativeInt = 0
    reload_policy_between_attempts: bool = True
    unlock_target_label: str | None = None
    start_manual_control: bool = False
    career_mode_race_setup: CareerModeRaceSetupConfig | None = None
    policy_observation_layout_shape_hint: tuple[PositiveInt, PositiveInt, PositiveInt] | None = None
    recording: WatchRecordingConfig = Field(default_factory=WatchRecordingConfig)
    career_debug: WatchCareerDebugConfig = Field(default_factory=WatchCareerDebugConfig)

    @model_validator(mode="after")
    def _default_split_fps(self) -> WatchConfig:
        if self.control_fps is None:
            self.control_fps = "auto"
        if self.render_fps is None:
            self.render_fps = 60.0
        if self.recording.enabled and self.recording.path is None:
            raise ValueError("watch.recording.path is required when recording is enabled")
        return self


class WatchAppConfig(BaseModel):
    """Top-level watch application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    track: TrackConfig = Field(default_factory=TrackConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    policy: PolicyConfig | None = None
    train: TrainConfig | None = None
    watch: WatchConfig = Field(default_factory=WatchConfig)


class TrainAppConfig(BaseModel):
    """Top-level train application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    track: TrackConfig = Field(default_factory=TrackConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _validate_recurrent_algorithm_alignment(self) -> TrainAppConfig:
        recurrent_enabled = self.policy.recurrent.enabled
        algorithm = self.train.algorithm
        if recurrent_enabled and algorithm not in TRAINING_ALGORITHMS.recurrent:
            raise ValueError("policy.recurrent.enabled=true requires a recurrent train.algorithm")
        if not recurrent_enabled and algorithm in TRAINING_ALGORITHMS.recurrent:
            raise ValueError(f"train.algorithm={algorithm} requires policy.recurrent.enabled=true")
        return self
