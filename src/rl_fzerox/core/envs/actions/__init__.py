# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.domain.action_adapters import (
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN_PRIMITIVE,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH,
    ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN,
    ACTION_ADAPTER_STEER_DRIVE,
    ACTION_ADAPTER_STEER_DRIVE_BOOST,
    ACTION_ADAPTER_STEER_DRIVE_BOOST_LEAN,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_LEAN,
    DEFAULT_ACTION_ADAPTER_NAME,
    ActionAdapterName,
)
from rl_fzerox.core.envs.actions.base import (
    ActionAdapter,
    ActionValue,
    DiscreteActionDimension,
    ResettableActionAdapter,
)
from rl_fzerox.core.envs.actions.continuous_steer_drive import (
    ContinuousSteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.hybrid_steer_drive import (
    HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerDriveBoostLeanActionAdapter,
    HybridSteerDriveBoostLeanPrimitiveActionAdapter,
    HybridSteerDriveLeanActionAdapter,
    HybridSteerGasAirBrakeBoostLeanActionAdapter,
    HybridSteerGasAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerGasBoostLeanActionAdapter,
)
from rl_fzerox.core.envs.actions.steer_drive import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BRAKE_MASK,
    THROTTLE_MASK,
    SteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.steer_drive_boost import SteerDriveBoostActionAdapter
from rl_fzerox.core.envs.actions.steer_drive_boost_lean import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    SteerDriveBoostLeanActionAdapter,
    SteerGasAirBrakeBoostLeanActionAdapter,
)

ActionAdapterFactory = Callable[[ActionRuntimeConfig], ActionAdapter]
DEFAULT_ACTION_NAME: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
ACTION_ADAPTER_REGISTRY: dict[ActionAdapterName, ActionAdapterFactory] = {
    ACTION_ADAPTER_CONTINUOUS_STEER_DRIVE: ContinuousSteerDriveActionAdapter,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN: HybridSteerDriveBoostLeanActionAdapter,
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_BOOST_LEAN_PRIMITIVE: (
        HybridSteerDriveBoostLeanPrimitiveActionAdapter
    ),
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH: (
        HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter
    ),
    ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN: HybridSteerGasBoostLeanActionAdapter,
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN: (
        HybridSteerGasAirBrakeBoostLeanActionAdapter
    ),
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH: (
        HybridSteerGasAirBrakeBoostLeanPitchActionAdapter
    ),
    ACTION_ADAPTER_HYBRID_STEER_DRIVE_LEAN: HybridSteerDriveLeanActionAdapter,
    ACTION_ADAPTER_STEER_GAS_AIR_BRAKE_BOOST_LEAN: SteerGasAirBrakeBoostLeanActionAdapter,
    ACTION_ADAPTER_STEER_DRIVE: SteerDriveActionAdapter,
    ACTION_ADAPTER_STEER_DRIVE_BOOST: SteerDriveBoostActionAdapter,
    ACTION_ADAPTER_STEER_DRIVE_BOOST_LEAN: SteerDriveBoostLeanActionAdapter,
}


def build_action_adapter(config: ActionConfig | ActionRuntimeConfig) -> ActionAdapter:
    """Construct one registered action adapter from the env config."""

    runtime_config = config.runtime() if isinstance(config, ActionConfig) else config
    factory = ACTION_ADAPTER_REGISTRY.get(runtime_config.name)
    if factory is None:
        raise ValueError(f"Unsupported action adapter: {runtime_config.name!r}")
    return factory(runtime_config)


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
    "DEFAULT_ACTION_NAME",
    "DiscreteActionDimension",
    "LEAN_LEFT_MASK",
    "LEAN_RIGHT_MASK",
    "HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter",
    "HybridSteerDriveBoostLeanActionAdapter",
    "HybridSteerDriveBoostLeanPrimitiveActionAdapter",
    "HybridSteerDriveLeanActionAdapter",
    "HybridSteerGasBoostLeanActionAdapter",
    "HybridSteerGasAirBrakeBoostLeanActionAdapter",
    "HybridSteerGasAirBrakeBoostLeanPitchActionAdapter",
    "ResettableActionAdapter",
    "SteerDriveActionAdapter",
    "SteerDriveBoostActionAdapter",
    "SteerDriveBoostLeanActionAdapter",
    "SteerGasAirBrakeBoostLeanActionAdapter",
    "THROTTLE_MASK",
    "action_adapter_names",
    "build_action_adapter",
]
