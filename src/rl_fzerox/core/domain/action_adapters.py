# src/rl_fzerox/core/domain/action_adapters.py
"""String-backed action adapter names shared by config, env factories, and validation."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

ActionAdapterName: TypeAlias = Literal[
    "continuous_steer_drive",
    "continuous_steer_drive_shoulder",
    "continuous_steer_drive_drift",
    "hybrid_steer_drive_boost_shoulder",
    "hybrid_steer_drive_boost_drift",
    "hybrid_steer_drive_boost_shoulder_primitive",
    "hybrid_steer_gas_air_brake_boost_shoulder",
    "hybrid_steer_drive_shoulder",
    "hybrid_steer_drive_drift",
    "steer_gas_air_brake_boost_shoulder",
    "steer_gas_air_brake_boost_drift",
    "steer_drive",
    "steer_drive_boost",
    "steer_drive_boost_shoulder",
    "steer_drive_boost_drift",
]

ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE: Final[ActionAdapterName] = "continuous_steer_drive"
ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_SHOULDER: Final[ActionAdapterName] = (
    "continuous_steer_drive_shoulder"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_shoulder"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_shoulder_primitive"
)
ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_SHOULDER: Final[ActionAdapterName] = (
    "hybrid_steer_gas_air_brake_boost_shoulder"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_SHOULDER: Final[ActionAdapterName] = "hybrid_steer_drive_shoulder"
ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_SHOULDER: Final[ActionAdapterName] = (
    "steer_gas_air_brake_boost_shoulder"
)
ACTION_ADAPTER_STEER_DRIVE: Final[ActionAdapterName] = "steer_drive"
ACTION_ADAPTER_STEER_DRIVE_BOOST: Final[ActionAdapterName] = "steer_drive_boost"
ACTION_ADAPTER_STEER_DRIVE_BOOST_SHOULDER: Final[ActionAdapterName] = "steer_drive_boost_shoulder"

# COMPAT SHIM: legacy action adapter names.
# Early configs used "drift" for F-Zero X's Z/R shoulder lean/slide inputs.
# Keep those string values loadable for historical run manifests.
ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT: Final[ActionAdapterName] = (
    "continuous_steer_drive_drift"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_drift"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT: Final[ActionAdapterName] = "hybrid_steer_drive_drift"
ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_DRIFT: Final[ActionAdapterName] = (
    "steer_gas_air_brake_boost_drift"
)
ACTION_ADAPTER_STEER_DRIVE_BOOST_DRIFT: Final[ActionAdapterName] = "steer_drive_boost_drift"

DEFAULT_ACTION_ADAPTER_NAME: Final[ActionAdapterName] = ACTION_ADAPTER_STEER_DRIVE_BOOST_SHOULDER


def _adapter_set(*names: ActionAdapterName) -> frozenset[ActionAdapterName]:
    """Build typed immutable groups while keeping YAML-facing values as strings."""
    return frozenset(names)


SAC_ACTION_ADAPTERS: Final = _adapter_set(
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_SHOULDER,
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT,
)
HYBRID_ACTION_ADAPTERS: Final = _adapter_set(
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_SHOULDER,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_SHOULDER,
)
