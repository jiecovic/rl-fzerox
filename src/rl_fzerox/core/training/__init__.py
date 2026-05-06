# src/rl_fzerox/core/training/__init__.py
"""Lazy training facade.

This package reexports common run helpers without importing the SB3 runner or
the native-backed run materialization path at package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
    from rl_fzerox.core.training.runs import (
        RunPaths,
        apply_train_run_to_watch_config,
        build_run_paths,
        ensure_run_dirs,
        load_train_run_config,
        resolve_latest_model_path,
        resolve_latest_policy_path,
        resolve_train_run_config_path,
        save_train_run_config,
    )


class _RunTrainingFn(Protocol):
    def __call__(self, config: TrainAppConfig) -> object: ...


_EXPORT_MODULES = {
    "RunPaths": "rl_fzerox.core.training.runs",
    "apply_train_run_to_watch_config": "rl_fzerox.core.training.runs",
    "build_run_paths": "rl_fzerox.core.training.runs",
    "ensure_run_dirs": "rl_fzerox.core.training.runs",
    "load_train_run_config": "rl_fzerox.core.training.runs",
    "resolve_latest_model_path": "rl_fzerox.core.training.runs",
    "resolve_latest_policy_path": "rl_fzerox.core.training.runs",
    "resolve_train_run_config_path": "rl_fzerox.core.training.runs",
    "save_train_run_config": "rl_fzerox.core.training.runs",
}


def run_training(config: TrainAppConfig) -> object:
    """Lazily import the SB3 training runner only when it is actually used."""

    return _load_run_training()(config)


def _load_run_training() -> _RunTrainingFn:
    return import_module("rl_fzerox.core.training.runner").run_training


__all__ = [
    "RunPaths",
    "apply_train_run_to_watch_config",
    "build_run_paths",
    "ensure_run_dirs",
    "load_train_run_config",
    "resolve_latest_model_path",
    "resolve_latest_policy_path",
    "resolve_train_run_config_path",
    "run_training",
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
