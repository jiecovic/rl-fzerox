# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import ActionAdapter, ActionValue
from rl_fzerox.core.envs.actions.steer_drive import (
    BRAKE_MASK,
    THROTTLE_MASK,
    SteerDriveActionAdapter,
)
from rl_fzerox.core.envs.actions.steer_drive_boost import SteerDriveBoostActionAdapter
from rl_fzerox.core.envs.actions.steer_drive_boost_drift import (
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    SteerDriveBoostDriftActionAdapter,
)

ActionAdapterFactory = Callable[[ActionConfig], ActionAdapter]
DEFAULT_ACTION_NAME = "steer_drive_boost_drift"
ACTION_ADAPTER_REGISTRY: dict[str, ActionAdapterFactory] = {
    "steer_drive": SteerDriveActionAdapter,
    "steer_drive_boost": SteerDriveBoostActionAdapter,
    "steer_drive_boost_drift": SteerDriveBoostDriftActionAdapter,
}


def build_action_adapter(config: ActionConfig) -> ActionAdapter:
    """Construct one registered action adapter from the env config."""

    factory = ACTION_ADAPTER_REGISTRY.get(config.name)
    if factory is None:
        raise ValueError(f"Unsupported action adapter: {config.name!r}")
    return factory(config)


def action_adapter_names() -> tuple[str, ...]:
    """Return the registered action adapter names in insertion order."""

    return tuple(ACTION_ADAPTER_REGISTRY)


__all__ = [
    "ACTION_ADAPTER_REGISTRY",
    "ActionAdapter",
    "ActionAdapterFactory",
    "ActionValue",
    "BRAKE_MASK",
    "BOOST_MASK",
    "DEFAULT_ACTION_NAME",
    "DRIFT_LEFT_MASK",
    "DRIFT_RIGHT_MASK",
    "SteerDriveActionAdapter",
    "SteerDriveBoostActionAdapter",
    "SteerDriveBoostDriftActionAdapter",
    "THROTTLE_MASK",
    "action_adapter_names",
    "build_action_adapter",
]
