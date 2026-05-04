# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.domain.action_adapters import (
    ACTION_ADAPTERS,
    DEFAULT_ACTION_ADAPTER_NAME,
    LEGACY_ACTION_ADAPTERS,
    ActionAdapterName,
)
from rl_fzerox.core.envs.actions.base import (
    ActionAdapter,
    ActionValue,
    DiscreteActionDimension,
    ResettableActionAdapter,
)
from rl_fzerox.core.envs.actions.configured import (
    ConfiguredDiscreteActionAdapter,
    ConfiguredHybridActionAdapter,
)
from rl_fzerox.core.envs.actions.continuous.steer_drive import (
    ContinuousSteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BRAKE_MASK,
    THROTTLE_MASK,
    SteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive_boost import (
    SteerDriveBoostActionAdapter,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive_boost_lean import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    SteerDriveBoostLeanActionAdapter,
    SteerGasAirBrakeBoostLeanActionAdapter,
)
from rl_fzerox.core.envs.actions.hybrid.facade import (
    HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerDriveBoostLeanActionAdapter,
    HybridSteerDriveBoostLeanPrimitiveActionAdapter,
    HybridSteerDriveLeanActionAdapter,
    HybridSteerGasAirBrakeBoostLeanActionAdapter,
    HybridSteerGasAirBrakeBoostLeanPitchActionAdapter,
    HybridSteerGasBoostLeanActionAdapter,
)

ActionAdapterFactory = Callable[[ActionRuntimeConfig], ActionAdapter]
DEFAULT_ACTION_NAME: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
ACTION_ADAPTER_REGISTRY: dict[ActionAdapterName, ActionAdapterFactory] = {
    ACTION_ADAPTERS.configured_discrete: ConfiguredDiscreteActionAdapter,
    ACTION_ADAPTERS.configured_hybrid: ConfiguredHybridActionAdapter,
    LEGACY_ACTION_ADAPTERS.continuous_steer_drive: ContinuousSteerDriveActionAdapter,
    LEGACY_ACTION_ADAPTERS.hybrid_steer_drive_boost_lean: HybridSteerDriveBoostLeanActionAdapter,
    LEGACY_ACTION_ADAPTERS.hybrid_steer_drive_boost_lean_primitive: (
        HybridSteerDriveBoostLeanPrimitiveActionAdapter
    ),
    LEGACY_ACTION_ADAPTERS.hybrid_steer_drive_air_brake_boost_lean_pitch: (
        HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter
    ),
    LEGACY_ACTION_ADAPTERS.hybrid_steer_gas_boost_lean: HybridSteerGasBoostLeanActionAdapter,
    LEGACY_ACTION_ADAPTERS.hybrid_steer_gas_air_brake_boost_lean: (
        HybridSteerGasAirBrakeBoostLeanActionAdapter
    ),
    LEGACY_ACTION_ADAPTERS.hybrid_steer_gas_air_brake_boost_lean_pitch: (
        HybridSteerGasAirBrakeBoostLeanPitchActionAdapter
    ),
    LEGACY_ACTION_ADAPTERS.hybrid_steer_drive_lean: HybridSteerDriveLeanActionAdapter,
    LEGACY_ACTION_ADAPTERS.steer_gas_air_brake_boost_lean: SteerGasAirBrakeBoostLeanActionAdapter,
    LEGACY_ACTION_ADAPTERS.steer_drive: SteerDriveActionAdapter,
    LEGACY_ACTION_ADAPTERS.steer_drive_boost: SteerDriveBoostActionAdapter,
    LEGACY_ACTION_ADAPTERS.steer_drive_boost_lean: SteerDriveBoostLeanActionAdapter,
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
    "ConfiguredDiscreteActionAdapter",
    "ConfiguredHybridActionAdapter",
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
