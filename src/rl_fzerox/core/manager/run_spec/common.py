# src/rl_fzerox/core/manager/run_spec/common.py
"""Shared literals and aliases used across manager config sections."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import PositiveInt

from rl_fzerox.core.domain.observation_image import (
    ObservationPresetName,
)
from rl_fzerox.core.domain.observation_image import (
    ObservationResizeFilter as DomainObservationResizeFilter,
)

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
RaceMode = Literal["time_attack", "gp_race"]
TrackSamplingMode = Literal["equal", "step_balanced"]
TrackPoolMode = Literal["built_in", "x_cup"]
VehicleSelectionMode = Literal["fixed", "pool"]
EngineSettingMode = Literal["fixed", "random_range"]
ActionAxisMode = Literal["continuous", "discrete"]
ActionDriveMode = Literal["pwm", "on_off"]
LeanOutputMode = Literal["three_way", "independent_buttons"]
ObservationPreset = ObservationPresetName
ObservationResizeFilter = DomainObservationResizeFilter
ConvProfile = Literal[
    "nature",
    "custom",
]
FeatureDim: TypeAlias = PositiveInt | Literal["auto"]
ActivationName = Literal["relu", "tanh", "gelu"]
