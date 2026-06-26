# src/rl_fzerox/core/runtime_spec/__init__.py
"""Public runtime-spec façade used by env, training, and manifest IO.

The run-manager-owned ``core.manager.run_spec`` package is the canonical
authoring surface. ``core.runtime_spec`` exposes the lower-level resolved
runtime schema that training, watch, and saved run manifests validate against.
"""

from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.roms import (
    FZeroXRomResolutionError,
    find_fzerox_rom_path,
    fzerox_default_rom_path,
    fzerox_rom_candidates,
    fzerox_rom_dir,
    resolve_fzerox_rom_path,
)
from rl_fzerox.core.runtime_spec.schema import (
    ActionMaskConfig,
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
    "FZeroXRomResolutionError",
    "find_fzerox_rom_path",
    "fzerox_default_rom_path",
    "fzerox_rom_candidates",
    "fzerox_rom_dir",
    "project_root_dir",
    "resolve_fzerox_rom_path",
]
