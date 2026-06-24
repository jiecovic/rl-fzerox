# src/rl_fzerox/core/runtime_spec/schema/common.py
from __future__ import annotations

from typing import Literal

from pydantic import PositiveFloat

from rl_fzerox.core.domain.courses import (
    DeficitBudgetDifficultyMetric as DomainDeficitBudgetDifficultyMetric,
)
from rl_fzerox.core.domain.courses import (
    RuntimeTrackSamplingMode,
)
from rl_fzerox.core.domain.observations import (
    ObservationPresetName as DomainObservationPresetName,
)

type WatchFpsSetting = PositiveFloat | Literal["auto", "unlimited"]
type TrackSamplingMode = RuntimeTrackSamplingMode
type DeficitBudgetDifficultyMetric = DomainDeficitBudgetDifficultyMetric
type ObservationPresetName = DomainObservationPresetName
type ObservationResizeFilter = Literal["nearest", "bilinear"]
type ActionMaskOverrides = dict[str, tuple[int, ...]]
type ContinuousAirBrakeMode = Literal["always", "disable_on_ground", "off"]
