# src/rl_fzerox/core/manager/run_spec/common.py
"""Shared literals and aliases used across manager config sections."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import PositiveInt

from rl_fzerox.core.domain.lean import LeanOutputMode as LeanOutputMode
from rl_fzerox.core.domain.observation_image import (
    ObservationPresetName,
)
from rl_fzerox.core.domain.observation_image import (
    ObservationResizeFilter as DomainObservationResizeFilter,
)
from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.policy.activations import ActivationName as ActivationName

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
RaceMode = Literal["time_attack", "gp_race"]
GpDifficulty = RaceDifficultyName
TrackSamplingMode = Literal[
    "equal",
    "step_balanced",
    # Hidden from new manager UI metadata, but still accepted for legacy saved
    # run configs. Remove after compatibility migration.
    "adaptive_step_balanced",
    "fixed_env",
    "deficit_budget",
]
DeficitBudgetDifficultyMetric = Literal["completion_ema", "finish_ema", "mixed"]
VehicleSelectionMode = Literal["fixed", "pool"]
EngineSettingMode = Literal["fixed", "random_range", "adaptive_tuner"]
EngineTunerBackend = Literal["bandit", "gaussian_process", "mlp_ensemble"]
EngineTunerObjective = Literal["finish_time", "episode_return", "completion", "finish_rate"]
ActionAxisMode = Literal["continuous", "discrete"]
ActionDriveMode = Literal["pwm", "on_off"]
ObservationPreset = ObservationPresetName
ObservationResizeFilter = DomainObservationResizeFilter
ConvProfile = Literal[
    "nature",
    "impala_small",
    "impala_large",
    "custom",
]
FeatureDim: TypeAlias = PositiveInt | Literal["auto"]
