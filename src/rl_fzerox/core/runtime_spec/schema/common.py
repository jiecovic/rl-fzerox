# src/rl_fzerox/core/runtime_spec/schema/common.py
from __future__ import annotations

from typing import Literal

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

type WatchFpsSetting = PositiveFloat | Literal["auto", "unlimited"]
type TrackSamplingMode = RuntimeTrackSamplingMode
type DeficitBudgetDifficultyMetric = DomainDeficitBudgetDifficultyMetric
type ObservationPresetName = DomainObservationPresetName
type ObservationResizeFilter = DomainObservationResizeFilter
type ActionMaskOverrides = dict[str, tuple[int, ...]]
type ContinuousAirBrakeMode = Literal["always", "disable_on_ground", "off"]
