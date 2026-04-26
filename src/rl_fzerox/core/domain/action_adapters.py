# src/rl_fzerox/core/domain/action_adapters.py
"""String-backed action adapter names shared by config, env factories, and validation."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

ActionAdapterName: TypeAlias = Literal[
    "continuous_steer_drive",
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

ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE: Final[ActionAdapterName] = "continuous_steer_drive"
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_lean"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN_PRIMITIVE: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_lean_primitive"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH: Final[ActionAdapterName] = (
    "hybrid_steer_drive_air_brake_boost_lean_pitch"
)
ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN: Final[ActionAdapterName] = "hybrid_steer_gas_boost_lean"
ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN: Final[ActionAdapterName] = (
    "hybrid_steer_gas_air_brake_boost_lean"
)
ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH: Final[ActionAdapterName] = (
    "hybrid_steer_gas_air_brake_boost_lean_pitch"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_LEAN: Final[ActionAdapterName] = "hybrid_steer_drive_lean"
ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_LEAN: Final[ActionAdapterName] = (
    "steer_gas_air_brake_boost_lean"
)
ACTION_ADAPTER_STEER_DRIVE: Final[ActionAdapterName] = "steer_drive"
ACTION_ADAPTER_STEER_DRIVE_BOOST: Final[ActionAdapterName] = "steer_drive_boost"
ACTION_ADAPTER_STEER_DRIVE_BOOST_LEAN: Final[ActionAdapterName] = "steer_drive_boost_lean"

DEFAULT_ACTION_ADAPTER_NAME: Final[ActionAdapterName] = ACTION_ADAPTER_STEER_DRIVE_BOOST_LEAN


def _adapter_set(*names: ActionAdapterName) -> frozenset[ActionAdapterName]:
    """Build typed immutable groups while keeping YAML-facing values as strings."""
    return frozenset(names)


SAC_ACTION_ADAPTERS: Final = _adapter_set(
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
)
HYBRID_ACTION_ADAPTERS: Final = _adapter_set(
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN_PRIMITIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH,
    ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH,
)

CONTINUOUS_DRIVE_ACTION_ADAPTERS: Final = _adapter_set(
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN_PRIMITIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH,
)
