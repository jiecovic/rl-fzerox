# src/rl_fzerox/core/manager/store_api/checkpoints.py
"""Published checkpoint methods mixed into the public ManagerStore facade."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.models import ManagedPublishedCheckpoint
from rl_fzerox.core.manager.registry import checkpoints as checkpoint_registry
from rl_fzerox.core.manager.store_api.common import manager_store as _manager_store


class CheckpointStoreMixin:
    """ManagerStore facade methods for installed checkpoint records."""

    def import_published_checkpoint_bundle(
        self,
        *,
        bundle_path: Path,
        target_root: Path | None = None,
    ) -> ManagedPublishedCheckpoint:
        return checkpoint_registry.import_published_checkpoint_bundle(
            _manager_store(self),
            bundle_path=bundle_path,
            target_root=target_root,
        )

    def get_published_checkpoint(
        self,
        checkpoint_id: str,
    ) -> ManagedPublishedCheckpoint | None:
        return checkpoint_registry.get_published_checkpoint(_manager_store(self), checkpoint_id)

    def list_published_checkpoints(self) -> tuple[ManagedPublishedCheckpoint, ...]:
        return checkpoint_registry.list_published_checkpoints(_manager_store(self))
