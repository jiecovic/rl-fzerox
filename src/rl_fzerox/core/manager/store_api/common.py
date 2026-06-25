# src/rl_fzerox/core/manager/store_api/common.py
"""Shared helpers for ManagerStore facade mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def manager_store(store: object) -> ManagerStore:
    """Narrow a mixin instance back to the concrete ManagerStore owner."""

    from rl_fzerox.core.manager.store import ManagerStore

    if not isinstance(store, ManagerStore):
        raise TypeError("store facade must be mixed into ManagerStore")
    return store
