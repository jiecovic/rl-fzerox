# src/rl_fzerox/core/training/__init__.py
from __future__ import annotations

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


def run_training(*args, **kwargs):
    """Lazily import the SB3 training runner only when it is actually used."""

    from rl_fzerox.core.training.runner import run_training as _run_training

    return _run_training(*args, **kwargs)


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
