# src/rl_fzerox/core/manager/artifacts/__init__.py
"""Disk-side helpers owned by the run manager.

This package covers managed-run paths, pinned fork snapshots, and persisted
filesystem operations that must survive process restarts.
"""

from rl_fzerox.core.manager.artifacts.filesystem import (
    FilesystemOperation,
    apply_filesystem_operation,
    filesystem_operation_from_row,
    queue_delete_tree,
    queue_move_tree,
)
from rl_fzerox.core.manager.artifacts.fork_source import (
    clone_fork_source,
    draft_fork_source_dir,
    reset_fork_source_dir,
    run_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.artifacts.paths import (
    manager_runs_root,
    predicted_managed_lineage_dir,
    predicted_managed_run_dir,
)

__all__ = [
    "FilesystemOperation",
    "apply_filesystem_operation",
    "clone_fork_source",
    "draft_fork_source_dir",
    "filesystem_operation_from_row",
    "manager_runs_root",
    "predicted_managed_lineage_dir",
    "predicted_managed_run_dir",
    "queue_delete_tree",
    "queue_move_tree",
    "reset_fork_source_dir",
    "run_fork_source_dir",
    "snapshot_fork_source",
]
