# src/rl_fzerox/core/config/schema_models/apps.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rl_fzerox.core.config.schema_models.common import WatchFpsSetting
from rl_fzerox.core.config.schema_models.curriculum import CurriculumConfig
from rl_fzerox.core.config.schema_models.env import EmulatorConfig, EnvConfig, RewardConfig
from rl_fzerox.core.config.schema_models.policy import PolicyConfig
from rl_fzerox.core.config.schema_models.tracks import TrackConfig
from rl_fzerox.core.config.schema_models.training import TrainConfig
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS


class WatchConfig(BaseModel):
    """Human-facing watch UI settings."""

    model_config = ConfigDict(extra="forbid")

    episodes: int | None = Field(default=None, gt=0)
    control_fps: WatchFpsSetting | None = None
    render_fps: WatchFpsSetting | None = None
    deterministic_policy: bool = True
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    policy_run_dir: Path | None = None
    policy_artifact: Literal["latest", "best", "final"] = "latest"

    @model_validator(mode="after")
    def _default_split_fps(self) -> WatchConfig:
        if self.control_fps is None:
            self.control_fps = "auto"
        if self.render_fps is None:
            self.render_fps = 60.0
        return self


class WatchAppConfig(BaseModel):
    """Top-level watch application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    track: TrackConfig = Field(default_factory=TrackConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
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
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
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
