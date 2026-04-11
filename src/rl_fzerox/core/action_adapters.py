# src/rl_fzerox/core/action_adapters.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias

ActionAdapterName: TypeAlias = Literal[
    "continuous_steer_drive",
    "continuous_steer_drive_drift",
    "hybrid_steer_drive_boost_drift",
    "hybrid_steer_drive_boost_shoulder_primitive",
    "hybrid_steer_drive_drift",
    "steer_drive",
    "steer_drive_boost",
    "steer_drive_boost_drift",
]

ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE: Final[ActionAdapterName] = "continuous_steer_drive"
ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT: Final[ActionAdapterName] = (
    "continuous_steer_drive_drift"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_drift"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE: Final[ActionAdapterName] = (
    "hybrid_steer_drive_boost_shoulder_primitive"
)
ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT: Final[ActionAdapterName] = (
    "hybrid_steer_drive_drift"
)
ACTION_ADAPTER_STEER_DRIVE: Final[ActionAdapterName] = "steer_drive"
ACTION_ADAPTER_STEER_DRIVE_BOOST: Final[ActionAdapterName] = "steer_drive_boost"
ACTION_ADAPTER_STEER_DRIVE_BOOST_DRIFT: Final[ActionAdapterName] = (
    "steer_drive_boost_drift"
)
DEFAULT_ACTION_ADAPTER_NAME: Final[ActionAdapterName] = ACTION_ADAPTER_STEER_DRIVE_BOOST_DRIFT

SAC_ACTION_ADAPTERS: Final[frozenset[ActionAdapterName]] = frozenset(
    {
        ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
        ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT,
    }
)
HYBRID_ACTION_ADAPTERS: Final[frozenset[ActionAdapterName]] = frozenset(
    {
        ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT,
        ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT,
        ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE,
    }
)
