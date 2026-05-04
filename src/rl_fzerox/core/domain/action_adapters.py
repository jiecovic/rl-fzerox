"""String-backed action adapter names shared by config, env factories, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from .action_adapters_legacy import LEGACY_ACTION_ADAPTERS

ActionAdapterName: TypeAlias = Literal[
    "continuous_steer_drive",
    "configured_discrete",
    "configured_hybrid",
    "hybrid_steer_drive_boost_lean",
    "hybrid_steer_drive_boost_lean_primitive",
    "hybrid_steer_drive_air_brake_boost_lean_pitch",
    "hybrid_steer_gas_boost_lean",
    "hybrid_steer_gas_air_brake_boost_lean",
    "hybrid_steer_gas_air_brake_boost_lean_pitch",
    "hybrid_steer_drive_lean",
    "steer_gas_air_brake_boost_lean",
    "steer_drive",
    "steer_drive_boost",
    "steer_drive_boost_lean",
]


@dataclass(frozen=True, slots=True)
class ActionAdapterCatalog:
    """Current compositional adapter names plus capability groups."""

    configured_discrete: ActionAdapterName = "configured_discrete"
    configured_hybrid: ActionAdapterName = "configured_hybrid"

    @property
    def default(self) -> ActionAdapterName:
        return LEGACY_ACTION_ADAPTERS.default

    @property
    def sac(self) -> frozenset[ActionAdapterName]:
        return LEGACY_ACTION_ADAPTERS.sac

    @property
    def hybrid(self) -> frozenset[ActionAdapterName]:
        return frozenset((self.configured_hybrid, *LEGACY_ACTION_ADAPTERS.hybrid))

    @property
    def continuous_drive(self) -> frozenset[ActionAdapterName]:
        return LEGACY_ACTION_ADAPTERS.continuous_drive


ACTION_ADAPTERS = ActionAdapterCatalog()
DEFAULT_ACTION_ADAPTER_NAME = ACTION_ADAPTERS.default
SAC_ACTION_ADAPTERS = ACTION_ADAPTERS.sac
HYBRID_ACTION_ADAPTERS = ACTION_ADAPTERS.hybrid
CONTINUOUS_DRIVE_ACTION_ADAPTERS = ACTION_ADAPTERS.continuous_drive

__all__ = [
    "ACTION_ADAPTERS",
    "ActionAdapterCatalog",
    "ActionAdapterName",
    "CONTINUOUS_DRIVE_ACTION_ADAPTERS",
    "DEFAULT_ACTION_ADAPTER_NAME",
    "HYBRID_ACTION_ADAPTERS",
    "LEGACY_ACTION_ADAPTERS",
    "SAC_ACTION_ADAPTERS",
]
