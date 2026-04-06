# src/rl_fzerox/core/config/models.py
from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
)


class ActionConfig(BaseModel):
    """Policy action adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    steer_buckets: int = Field(default=5, ge=3)


class EnvConfig(BaseModel):
    """Environment-level rollout settings that affect frame stepping."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 2
    reset_to_race: bool = False
    action: ActionConfig = Field(default_factory=ActionConfig)


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None


class WatchConfig(BaseModel):
    """Human-facing watch UI settings."""

    model_config = ConfigDict(extra="forbid")

    episodes: PositiveInt | None = None
    fps: PositiveFloat | None = None


class WatchAppConfig(BaseModel):
    """Top-level watch application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    env: EnvConfig = Field(default_factory=EnvConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
