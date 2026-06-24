# src/rl_fzerox/core/manager/run_spec/common.py
"""Shared literals and aliases used across manager config sections."""

from __future__ import annotations

from typing import Literal

from pydantic import PositiveInt

from rl_fzerox.core.domain.courses import (
    DeficitBudgetDifficultyMetric as DeficitBudgetDifficultyMetric,
)
from rl_fzerox.core.domain.courses import (
    ManagedTrackSamplingMode,
)
from rl_fzerox.core.domain.engine import (
    EngineTunerBackend as EngineTunerBackend,
)
from rl_fzerox.core.domain.engine import (
    EngineTunerObjective as EngineTunerObjective,
)
from rl_fzerox.core.domain.observations import (
    ObservationPresetName,
)
from rl_fzerox.core.domain.race import LeanOutputMode as LeanOutputMode
from rl_fzerox.core.domain.race import RaceDifficultyName
from rl_fzerox.core.policy.activations import ActivationName as ActivationName
from rl_fzerox.core.runtime_spec.schema.common import (
    ObservationResizeFilter as ObservationResizeFilter,
)

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
RaceMode = Literal["time_attack", "gp_race"]
GpDifficulty = RaceDifficultyName
TrackSamplingMode = ManagedTrackSamplingMode
VehicleSelectionMode = Literal["fixed", "pool"]
EngineSettingMode = Literal["fixed", "random_range", "adaptive_tuner"]
ActionAxisMode = Literal["continuous", "discrete"]
ActionDriveMode = Literal["pwm", "on_off"]
ObservationPreset = ObservationPresetName
ConvProfile = Literal[
    "nature",
    "impala_small",
    "impala_large",
    "custom",
]
type FeatureDim = PositiveInt | Literal["auto"]
