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
from rl_fzerox.core.domain.track_sampling import (
    DeficitBudgetDifficultyMetric as DomainDeficitBudgetDifficultyMetric,
)
from rl_fzerox.core.domain.track_sampling import (
    RuntimeTrackSamplingMode,
)

WatchFpsSetting: TypeAlias = PositiveFloat | Literal["auto", "unlimited"]
TrackSamplingMode: TypeAlias = RuntimeTrackSamplingMode
DeficitBudgetDifficultyMetric: TypeAlias = DomainDeficitBudgetDifficultyMetric
ObservationPresetName: TypeAlias = DomainObservationPresetName
ObservationResizeFilter: TypeAlias = DomainObservationResizeFilter
ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]
ContinuousAirBrakeMode: TypeAlias = Literal["always", "disable_on_ground", "off"]
