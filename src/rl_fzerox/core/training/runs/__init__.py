# src/rl_fzerox/core/training/runs/__init__.py
"""Lazy training-run facade.

Expose common run-path and config helpers without pulling the config
materializer or emulator-facing baseline pipeline into import-time paths that
only need filesystem helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.training.runs.artifacts import (
        resolve_latest_model_path,
        resolve_latest_policy_path,
        resolve_model_artifact_path,
        resolve_policy_artifact_path,
    )
    from rl_fzerox.core.training.runs.config import (
        apply_train_run_to_watch_config,
        load_train_run_config,
        load_train_run_config_for_watch,
        load_train_run_train_config,
        materialize_train_run_config,
        materialize_watch_session_config,
        save_train_run_config,
    )
    from rl_fzerox.core.training.runs.paths import (
        RUN_LAYOUT,
        RunPaths,
        WatchSessionPaths,
        build_run_paths,
        build_watch_session_paths,
        continue_run_paths,
        ensure_run_dirs,
        ensure_watch_session_dirs,
        explicit_run_paths,
        reserve_run_paths,
        resolve_train_run_config_path,
    )

_EXPORT_MODULES = {
    "resolve_latest_model_path": "rl_fzerox.core.training.runs.artifacts",
    "resolve_latest_policy_path": "rl_fzerox.core.training.runs.artifacts",
    "resolve_model_artifact_path": "rl_fzerox.core.training.runs.artifacts",
    "resolve_policy_artifact_path": "rl_fzerox.core.training.runs.artifacts",
    "apply_train_run_to_watch_config": "rl_fzerox.core.training.runs.config",
    "load_train_run_config": "rl_fzerox.core.training.runs.config",
    "load_train_run_config_for_watch": "rl_fzerox.core.training.runs.config",
    "load_train_run_train_config": "rl_fzerox.core.training.runs.config",
    "materialize_train_run_config": "rl_fzerox.core.training.runs.config",
    "materialize_watch_session_config": "rl_fzerox.core.training.runs.config",
    "save_train_run_config": "rl_fzerox.core.training.runs.config",
    "RUN_LAYOUT": "rl_fzerox.core.training.runs.paths",
    "RunPaths": "rl_fzerox.core.training.runs.paths",
    "WatchSessionPaths": "rl_fzerox.core.training.runs.paths",
    "build_run_paths": "rl_fzerox.core.training.runs.paths",
    "build_watch_session_paths": "rl_fzerox.core.training.runs.paths",
    "continue_run_paths": "rl_fzerox.core.training.runs.paths",
    "ensure_run_dirs": "rl_fzerox.core.training.runs.paths",
    "ensure_watch_session_dirs": "rl_fzerox.core.training.runs.paths",
    "explicit_run_paths": "rl_fzerox.core.training.runs.paths",
    "reserve_run_paths": "rl_fzerox.core.training.runs.paths",
    "resolve_train_run_config_path": "rl_fzerox.core.training.runs.paths",
}

__all__ = [
    "RUN_LAYOUT",
    "RunPaths",
    "WatchSessionPaths",
    "apply_train_run_to_watch_config",
    "build_run_paths",
    "build_watch_session_paths",
    "continue_run_paths",
    "ensure_run_dirs",
    "ensure_watch_session_dirs",
    "explicit_run_paths",
    "load_train_run_config",
    "load_train_run_config_for_watch",
    "load_train_run_train_config",
    "materialize_train_run_config",
    "materialize_watch_session_config",
    "resolve_latest_model_path",
    "resolve_latest_policy_path",
    "resolve_model_artifact_path",
    "resolve_policy_artifact_path",
    "resolve_train_run_config_path",
    "reserve_run_paths",
    "save_train_run_config",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
