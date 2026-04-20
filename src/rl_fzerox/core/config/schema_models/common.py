# src/rl_fzerox/core/config/schema_models/common.py
from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import PositiveFloat

WatchFpsSetting: TypeAlias = PositiveFloat | Literal["auto", "unlimited"]
TrackSamplingMode: TypeAlias = Literal["random", "balanced"]
ObservationPresetName: TypeAlias = Literal[
    "crop_84x116",
    "crop_92x124",
    "crop_116x164",
    "crop_98x130",
    "crop_66x82",
]
ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]
ContinuousDriveMode: TypeAlias = Literal["threshold", "pwm", "always_accelerate"]
ContinuousAirBrakeMode: TypeAlias = Literal["always", "disable_on_ground", "off"]

LEGACY_OBSERVATION_PRESET_ALIASES: dict[str, ObservationPresetName] = {
    "native_crop_v1": "crop_84x116",
    "native_crop_v2": "crop_92x124",
    "native_crop_v3": "crop_116x164",
    "native_crop_v4": "crop_98x130",
    "native_crop_v6": "crop_66x82",
}
