# src/rl_fzerox/core/manager/registry/tensorboard.py
"""Registry facade for rebuilding TensorBoard lineage views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.artifacts.tensorboard_views import (
    TensorboardViewGroup,
)
from rl_fzerox.core.manager.artifacts.tensorboard_views import (
    rebuild_tensorboard_views as rebuild_tensorboard_view_artifacts,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def rebuild_tensorboard_views(store: ManagerStore) -> tuple[TensorboardViewGroup, ...]:
    return rebuild_tensorboard_view_artifacts(
        store.list_visible_runs(),
        view_root=store.tensorboard_views_root(),
    )
