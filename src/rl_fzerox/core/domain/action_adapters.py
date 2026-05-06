"""String-backed action adapter names shared by config, env factories, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ActionAdapterName: TypeAlias = Literal[
    "configured_discrete",
    "configured_hybrid",
]


@dataclass(frozen=True, slots=True)
class ActionAdapterCatalog:
    """Configured runtime action layouts supported by the managed surface."""

    configured_discrete: ActionAdapterName = "configured_discrete"
    configured_hybrid: ActionAdapterName = "configured_hybrid"

    @property
    def default(self) -> ActionAdapterName:
        return self.configured_discrete


ACTION_ADAPTERS = ActionAdapterCatalog()
DEFAULT_ACTION_ADAPTER_NAME = ACTION_ADAPTERS.default

__all__ = [
    "ACTION_ADAPTERS",
    "ActionAdapterCatalog",
    "ActionAdapterName",
    "DEFAULT_ACTION_ADAPTER_NAME",
]
