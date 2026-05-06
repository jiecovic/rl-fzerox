"""Shared literals and aliases used across manager config sections."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import PositiveInt

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
ObservationPreset = Literal[
    "crop_84x116",
    "crop_92x124",
    "crop_116x164",
    "crop_98x130",
    "crop_66x82",
    "crop_60x76",
    "crop_68x68",
    "crop_84x84",
    "crop_76x100",
    "crop_64x64",
]
ObservationResizeFilter = Literal["nearest", "bilinear"]
ConvProfile = Literal[
    "auto",
    "nature",
    "nature_32_64_128",
    "nature_wide",
    "nature_extra_k3",
    "compact_deep",
    "compact_bottleneck",
    "tiny_256",
    "custom",
]
FeatureDim: TypeAlias = PositiveInt | Literal["auto"]
ActivationName = Literal["relu", "tanh", "gelu"]
