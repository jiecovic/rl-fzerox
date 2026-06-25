# src/rl_fzerox/core/manager/registry/drafts/fork_sources.py
"""Filesystem snapshot handling for draft fork sources."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.artifacts.fork_source import (
    draft_fork_source_dir,
    reset_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.models import ManagedRun


def snapshot_draft_source(
    *,
    manager_db_path: Path,
    draft_id: str,
    source_run: ManagedRun,
    source_artifact: Literal["latest", "best"],
) -> tuple[Path, int]:
    destination_dir = draft_fork_source_dir(
        manager_db_path=manager_db_path,
        draft_id=draft_id,
    )
    source_num_timesteps = snapshot_fork_source(
        source_run_dir=source_run.run_dir,
        artifact=source_artifact,
        destination_dir=destination_dir,
    )
    return destination_dir, source_num_timesteps


def reset_draft_source(snapshot_dir: Path | None) -> None:
    if snapshot_dir is None:
        return
    reset_fork_source_dir(snapshot_dir)
