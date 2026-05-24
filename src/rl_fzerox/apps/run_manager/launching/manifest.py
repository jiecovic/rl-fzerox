# src/rl_fzerox/apps/run_manager/launching/manifest.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.runs import save_train_run_config


def persist_launch_manifest(*, run_dir: Path, train_config: TrainAppConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_train_run_config(config=train_config, run_dir=run_dir)


def default_fork_name(source_name: str, artifact: Literal["latest", "best"]) -> str:
    suffix = "best fork" if artifact == "best" else "fork"
    return f"{source_name} {suffix}"
