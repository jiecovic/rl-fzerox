# src/rl_fzerox/core/camera.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

CameraSettingName: TypeAlias = Literal["overhead", "close_behind", "regular", "wide"]


@dataclass(frozen=True, slots=True)
class CameraSetting:
    """Camera mode name used by config plus the raw value reported by telemetry."""

    name: CameraSettingName
    raw: int


CAMERA_SETTINGS: Final[tuple[CameraSetting, ...]] = (
    CameraSetting("overhead", 0),
    CameraSetting("close_behind", 1),
    CameraSetting("regular", 2),
    CameraSetting("wide", 3),
)

CAMERA_SETTING_BY_NAME: Final[Mapping[CameraSettingName, CameraSetting]] = MappingProxyType(
    {setting.name: setting for setting in CAMERA_SETTINGS}
)
