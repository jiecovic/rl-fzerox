# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.domain.action_adapters import (
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT,
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_SHOULDER,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_SHOULDER,
    ACTION_ADAPTER_STEER_DRIVE,
    ACTION_ADAPTER_STEER_DRIVE_BOOST,
    ACTION_ADAPTER_STEER_DRIVE_BOOST_DRIFT,
    ACTION_ADAPTER_STEER_DRIVE_BOOST_SHOULDER,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_DRIFT,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_SHOULDER,
    DEFAULT_ACTION_ADAPTER_NAME,
    ActionAdapterName,
)
from rl_fzerox.core.envs.actions.base import ActionAdapter, ActionValue, ResettableActionAdapter
from rl_fzerox.core.envs.actions.continuous_steer_drive import (
    ContinuousSteerDriveActionAdapter,
    ContinuousSteerDriveDriftActionAdapter,
    ContinuousSteerDriveShoulderActionAdapter,
)
from rl_fzerox.core.envs.actions.hybrid_steer_drive import (
    HybridSteerDriveBoostDriftActionAdapter,
    HybridSteerDriveBoostShoulderActionAdapter,
    HybridSteerDriveBoostShoulderPrimitiveActionAdapter,
    HybridSteerDriveDriftActionAdapter,
    HybridSteerDriveShoulderActionAdapter,
)
from rl_fzerox.core.envs.actions.steer_drive import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BRAKE_MASK,
    THROTTLE_MASK,
    SteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.steer_drive_boost import SteerDriveBoostActionAdapter
from rl_fzerox.core.envs.actions.steer_drive_boost_drift import (
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    SHOULDER_LEFT_MASK,
    SHOULDER_RIGHT_MASK,
    SteerDriveBoostDriftActionAdapter,
    SteerDriveBoostShoulderActionAdapter,
    SteerGasAirBrakeBoostDriftActionAdapter,
    SteerGasAirBrakeBoostShoulderActionAdapter,
)

ActionAdapterFactory = Callable[[ActionConfig], ActionAdapter]
DEFAULT_ACTION_NAME: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
ACTION_ADAPTER_REGISTRY: dict[ActionAdapterName, ActionAdapterFactory] = {
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE: ContinuousSteerDriveActionAdapter,
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_SHOULDER: ContinuousSteerDriveShoulderActionAdapter,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER: (HybridSteerDriveBoostShoulderActionAdapter),
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_SHOULDER_PRIMITIVE: (
        HybridSteerDriveBoostShoulderPrimitiveActionAdapter
    ),
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_SHOULDER: HybridSteerDriveShoulderActionAdapter,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_SHOULDER: (SteerGasAirBrakeBoostShoulderActionAdapter),
    ACTION_ADAPTER_STEER_DRIVE: SteerDriveActionAdapter,
    ACTION_ADAPTER_STEER_DRIVE_BOOST: SteerDriveBoostActionAdapter,
    ACTION_ADAPTER_STEER_DRIVE_BOOST_SHOULDER: SteerDriveBoostShoulderActionAdapter,
    # COMPAT SHIM: legacy adapter names from saved run manifests.
    # Old "drift" names target the same Z/R shoulder-input adapters.
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE_DRIFT: ContinuousSteerDriveShoulderActionAdapter,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_DRIFT: (HybridSteerDriveBoostShoulderActionAdapter),
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_DRIFT: HybridSteerDriveShoulderActionAdapter,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_DRIFT: (SteerGasAirBrakeBoostShoulderActionAdapter),
    ACTION_ADAPTER_STEER_DRIVE_BOOST_DRIFT: SteerDriveBoostShoulderActionAdapter,
}


def build_action_adapter(config: ActionConfig) -> ActionAdapter:
    """Construct one registered action adapter from the env config."""

    factory = ACTION_ADAPTER_REGISTRY.get(config.name)
    if factory is None:
        raise ValueError(f"Unsupported action adapter: {config.name!r}")
    return factory(config)


def action_adapter_names() -> tuple[ActionAdapterName, ...]:
    """Return the registered action adapter names in insertion order."""

    return tuple(ACTION_ADAPTER_REGISTRY)


__all__ = [
    "ACTION_ADAPTER_REGISTRY",
    "ACCELERATE_MASK",
    "AIR_BRAKE_MASK",
    "ActionAdapter",
    "ActionAdapterFactory",
    "ActionValue",
    "BRAKE_MASK",
    "BOOST_MASK",
    "ContinuousSteerDriveActionAdapter",
    "ContinuousSteerDriveDriftActionAdapter",
    "ContinuousSteerDriveShoulderActionAdapter",
    "DEFAULT_ACTION_NAME",
    "DRIFT_LEFT_MASK",
    "DRIFT_RIGHT_MASK",
    "HybridSteerDriveBoostDriftActionAdapter",
    "HybridSteerDriveBoostShoulderActionAdapter",
    "HybridSteerDriveBoostShoulderPrimitiveActionAdapter",
    "HybridSteerDriveDriftActionAdapter",
    "HybridSteerDriveShoulderActionAdapter",
    "ResettableActionAdapter",
    "SHOULDER_LEFT_MASK",
    "SHOULDER_RIGHT_MASK",
    "SteerDriveActionAdapter",
    "SteerDriveBoostActionAdapter",
    "SteerDriveBoostDriftActionAdapter",
    "SteerDriveBoostShoulderActionAdapter",
    "SteerGasAirBrakeBoostDriftActionAdapter",
    "SteerGasAirBrakeBoostShoulderActionAdapter",
    "THROTTLE_MASK",
    "action_adapter_names",
    "build_action_adapter",
]
