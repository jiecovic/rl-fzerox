# src/rl_fzerox/core/manager/paths.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.config.paths import config_root_dir
from rl_fzerox.core.config.schema import TrainAppConfig


def manager_runs_root(*, output_root: Path | None = None) -> Path:
    """Return the manager-owned run root under the normal training output tree."""

    return (output_root or _base_train_config().train.output_root).resolve()


def predicted_managed_lineage_dir(
    lineage_id: str,
    *,
    output_root: Path | None = None,
) -> Path:
    """Return the exact lineage directory for one manager lineage id."""

    return manager_runs_root(output_root=output_root) / lineage_id


def predicted_managed_run_dir(
    run_id: str,
    *,
    lineage_id: str,
    output_root: Path | None = None,
) -> Path:
    """Return the exact manager-owned run directory for one run inside one lineage."""

    return predicted_managed_lineage_dir(lineage_id, output_root=output_root) / run_id


@lru_cache(maxsize=1)
def _base_train_config() -> TrainAppConfig:
    return load_train_app_config(config_root_dir() / "train_base.yaml")
