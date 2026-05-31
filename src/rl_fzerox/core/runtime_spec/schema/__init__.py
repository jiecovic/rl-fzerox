# src/rl_fzerox/core/runtime_spec/schema/__init__.py
"""Derived runtime schema package.

This package groups the resolved config models consumed by env, training, watch,
and saved manifests. It is intentionally separate from the manager-owned
``run_spec`` authoring model, which compiles into this lower-level runtime
shape through ``core.manager.projection``.
"""

from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema.actions import (
    ActionConfig,
    ActionMaskConfig,
    ActionRuntimeConfig,
)
from rl_fzerox.core.runtime_spec.schema.apps import (
    TrainAppConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.runtime_spec.schema.common import (
    ActionMaskOverrides,
    ContinuousAirBrakeMode,
    ObservationPresetName,
    TrackSamplingMode,
    WatchFpsSetting,
)
from rl_fzerox.core.runtime_spec.schema.curriculum import (
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
    CurriculumTriggerConfig,
    PerTrackLapsCompletedTriggerConfig,
)
from rl_fzerox.core.runtime_spec.schema.env import (
    EmulatorConfig,
    EnvConfig,
    RewardConfig,
    RewardCourseOverrideConfig,
)
from rl_fzerox.core.runtime_spec.schema.observations import (
    ObservationConfig,
    ObservationStateComponentConfig,
)
from rl_fzerox.core.runtime_spec.schema.policy import (
    ExtractorConfig,
    NetArchConfig,
    PolicyActionBiasConfig,
    PolicyAuxiliaryStateConfig,
    PolicyAuxiliaryStateLossConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
)
from rl_fzerox.core.runtime_spec.schema.tracks import (
    TrackConfig,
    TrackRecordEntryConfig,
    TrackRecordsConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    XCupRotationConfig,
)
from rl_fzerox.core.runtime_spec.schema.training import (
    StateFeatureDropoutGroupConfig,
    TrainActorRegularizationConfig,
    TrainConfig,
)

__all__ = [
    "ActionConfig",
    "ActionMaskConfig",
    "ActionMaskOverrides",
    "ActionRuntimeConfig",
    "ContinuousAirBrakeMode",
    "CurriculumConfig",
    "CurriculumStageConfig",
    "CurriculumTrainOverridesConfig",
    "CurriculumTriggerConfig",
    "EmulatorConfig",
    "EnvConfig",
    "ExtractorConfig",
    "NetArchConfig",
    "ObservationConfig",
    "ObservationPresetName",
    "ObservationStateComponentConfig",
    "PerTrackLapsCompletedTriggerConfig",
    "PolicyActionBiasConfig",
    "PolicyAuxiliaryStateConfig",
    "PolicyAuxiliaryStateLossConfig",
    "PolicyConfig",
    "PolicyRecurrentConfig",
    "RewardConfig",
    "RewardCourseOverrideConfig",
    "StateFeatureDropoutGroupConfig",
    "TrackConfig",
    "TrackRecordEntryConfig",
    "TrackRecordsConfig",
    "TrackSamplingConfig",
    "TrackSamplingEntryConfig",
    "TrackSamplingMode",
    "TrainActorRegularizationConfig",
    "TrainAppConfig",
    "TrainConfig",
    "WatchAppConfig",
    "WatchConfig",
    "WatchFpsSetting",
    "XCupRotationConfig",
]
