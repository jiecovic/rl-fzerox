# src/rl_fzerox/core/manager/registry/paths.py
"""Path resolution helpers for manager-owned local artifacts."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.artifacts.paths import (
    manager_evaluations_root as artifact_manager_evaluations_root,
)
from rl_fzerox.core.manager.artifacts.paths import (
    manager_runs_root as artifact_manager_runs_root,
)
from rl_fzerox.core.manager.artifacts.paths import (
    manager_save_games_root as artifact_manager_save_games_root,
)
from rl_fzerox.core.manager.artifacts.paths import (
    manager_tensorboard_views_root as artifact_manager_tensorboard_views_root,
)
from rl_fzerox.core.manager.artifacts.paths import (
    predicted_managed_run_dir,
)


def manager_root(*, output_root: Path | None = None) -> Path:
    return artifact_manager_runs_root(output_root=output_root)


def tensorboard_views_root(db_path: Path, *, output_root: Path | None = None) -> Path:
    if output_root is None:
        return db_path.parent.parent / "tensorboard_views"
    return artifact_manager_tensorboard_views_root(output_root=output_root)


def save_games_root(db_path: Path, *, output_root: Path | None = None) -> Path:
    if output_root is None:
        return db_path.parent.parent / "save_games"
    return artifact_manager_save_games_root(output_root=output_root)


def evaluations_root(db_path: Path, *, output_root: Path | None = None) -> Path:
    if output_root is None:
        return db_path.parent.parent / "evaluations"
    return artifact_manager_evaluations_root(output_root=output_root)


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def manager_run_dir(*, run_id: str, lineage_id: str, output_root: Path | None = None) -> Path:
    return predicted_managed_run_dir(run_id, lineage_id=lineage_id, output_root=output_root)
