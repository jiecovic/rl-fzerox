# src/rl_fzerox/core/manager/registry/paths.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.artifacts.paths import (
    manager_runs_root,
    predicted_managed_run_dir,
)


def manager_root(*, output_root: Path | None = None) -> Path:
    return manager_runs_root(output_root=output_root)


def manager_run_dir(*, run_id: str, lineage_id: str, output_root: Path | None = None) -> Path:
    return predicted_managed_run_dir(run_id, lineage_id=lineage_id, output_root=output_root)
