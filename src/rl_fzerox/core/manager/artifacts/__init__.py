# src/rl_fzerox/core/manager/artifacts/__init__.py
"""Disk-side helpers owned by the run manager.

This package covers managed-run paths, pinned fork snapshots, and persisted
filesystem operations that must survive process restarts.
"""

from rl_fzerox.core.manager.artifacts.filesystem import (
    FilesystemOperation,
    apply_filesystem_operation,
    filesystem_operation_from_values,
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
    manager_tensorboard_views_root,
    predicted_managed_lineage_dir,
    predicted_managed_run_dir,
)
from rl_fzerox.core.manager.artifacts.tensorboard_views import (
    TensorboardViewGroup,
    rebuild_tensorboard_views,
    slugify_path_segment,
)

__all__ = [
    "FilesystemOperation",
    "TensorboardViewGroup",
    "apply_filesystem_operation",
    "clone_fork_source",
    "draft_fork_source_dir",
    "filesystem_operation_from_values",
    "manager_runs_root",
    "manager_tensorboard_views_root",
    "predicted_managed_lineage_dir",
    "predicted_managed_run_dir",
    "rebuild_tensorboard_views",
    "reset_fork_source_dir",
    "run_fork_source_dir",
    "slugify_path_segment",
    "snapshot_fork_source",
]
