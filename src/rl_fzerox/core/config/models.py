# src/rl_fzerox/core/config/models.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, FilePath, PositiveFloat, PositiveInt


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 2


class EmulatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath


class WatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: PositiveInt = 1
    fps: PositiveFloat | None = None


class WatchAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    env: EnvConfig = Field(default_factory=EnvConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
