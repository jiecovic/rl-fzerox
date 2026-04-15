# src/rl_fzerox/core/config/__init__.py
from rl_fzerox.core.config.loader import load_train_app_config, load_watch_app_config
from rl_fzerox.core.config.paths import config_root_dir, project_root_dir
from rl_fzerox.core.config.schema import (
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
    "TrainAppConfig",
    "TrainConfig",
    "WatchAppConfig",
    "WatchConfig",
    "config_root_dir",
    "load_train_app_config",
    "load_watch_app_config",
    "project_root_dir",
]
