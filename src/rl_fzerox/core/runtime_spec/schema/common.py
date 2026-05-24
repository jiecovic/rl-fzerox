# src/rl_fzerox/core/runtime_spec/schema/common.py
from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import PositiveFloat

from rl_fzerox.core.domain.observation_image import (
    ObservationPresetName as DomainObservationPresetName,
)
from rl_fzerox.core.domain.observation_image import (
    ObservationResizeFilter as DomainObservationResizeFilter,
)

WatchFpsSetting: TypeAlias = PositiveFloat | Literal["auto", "unlimited"]
TrackSamplingMode: TypeAlias = Literal[
    "random",
    "balanced",
    "step_balanced",
    "adaptive_step_balanced",
]
ObservationPresetName: TypeAlias = DomainObservationPresetName
ObservationResizeFilter: TypeAlias = DomainObservationResizeFilter
ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]
ContinuousAirBrakeMode: TypeAlias = Literal["always", "disable_on_ground", "off"]
