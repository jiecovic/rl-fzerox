"""Public runtime-spec façade used by env, training, and manifest IO.

The run-manager-owned ``core.manager.run_spec`` package is the canonical
authoring surface. ``core.runtime_spec`` exposes the lower-level resolved
runtime schema that training, watch, and saved run manifests validate against.
"""

from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
    CurriculumTriggerConfig,
    EmulatorConfig,
    EnvConfig,
    ExtractorConfig,
    NetArchConfig,
    PolicyConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)

__all__ = [
    "ActionMaskConfig",
    "CurriculumConfig",
    "CurriculumStageConfig",
    "CurriculumTrainOverridesConfig",
    "CurriculumTriggerConfig",
    "EmulatorConfig",
    "EnvConfig",
    "ExtractorConfig",
    "NetArchConfig",
    "PolicyConfig",
    "TrackConfig",
    "TrackSamplingConfig",
    "TrackSamplingEntryConfig",
    "TrainAppConfig",
    "TrainConfig",
    "WatchAppConfig",
    "WatchConfig",
    "project_root_dir",
]
