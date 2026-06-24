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
    CareerModeRaceSetupConfig,
    TrainAppConfig,
    WatchAppConfig,
    WatchConfig,
    WatchRecordingConfig,
)
from rl_fzerox.core.runtime_spec.schema.common import (
    ActionMaskOverrides,
    ContinuousAirBrakeMode,
    ObservationPresetName,
    TrackSamplingMode,
    WatchFpsSetting,
)
from rl_fzerox.core.runtime_spec.schema.env import (
    EmulatorConfig,
    EnvConfig,
    RewardConfig,
    RewardCourseOverrideConfig,
)
from rl_fzerox.core.runtime_spec.schema.observations import (
    CustomResolutionChoice,
    ObservationConfig,
    ObservationResolutionConfig,
    ObservationResolutionMode,
    ObservationStateComponentConfig,
    PresetResolutionChoice,
    SourceCropResolutionChoice,
    resolve_observation_geometry,
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
    AdaptiveEngineTuningConfig,
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
    "AdaptiveEngineTuningConfig",
    "ContinuousAirBrakeMode",
    "CareerModeRaceSetupConfig",
    "CustomResolutionChoice",
    "EmulatorConfig",
    "EnvConfig",
    "ExtractorConfig",
    "NetArchConfig",
    "ObservationConfig",
    "ObservationPresetName",
    "ObservationResolutionConfig",
    "ObservationResolutionMode",
    "ObservationStateComponentConfig",
    "PolicyActionBiasConfig",
    "PolicyAuxiliaryStateConfig",
    "PolicyAuxiliaryStateLossConfig",
    "PolicyConfig",
    "PolicyRecurrentConfig",
    "RewardConfig",
    "RewardCourseOverrideConfig",
    "PresetResolutionChoice",
    "SourceCropResolutionChoice",
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
    "WatchRecordingConfig",
    "XCupRotationConfig",
    "resolve_observation_geometry",
]
