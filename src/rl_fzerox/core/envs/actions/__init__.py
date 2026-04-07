# src/rl_fzerox/core/envs/actions/__init__.py
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import ActionAdapter, ActionValue
from rl_fzerox.core.envs.actions.steer_drive import THROTTLE_MASK, SteerDriveActionAdapter
from rl_fzerox.core.envs.actions.steer_drive_boost_drift import (
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    SteerDriveBoostDriftActionAdapter,
)


def resolve_action_adapter(config: ActionConfig) -> ActionAdapter:
    """Construct the configured action adapter for one env instance."""

    if config.name == "steer_drive":
        return SteerDriveActionAdapter(config)
    if config.name == "steer_drive_boost_drift":
        return SteerDriveBoostDriftActionAdapter(config)
    raise ValueError(f"Unsupported action adapter: {config.name!r}")

__all__ = [
    "ActionAdapter",
    "ActionValue",
    "BOOST_MASK",
    "DRIFT_LEFT_MASK",
    "DRIFT_RIGHT_MASK",
    "SteerDriveActionAdapter",
    "SteerDriveBoostDriftActionAdapter",
    "THROTTLE_MASK",
    "resolve_action_adapter",
]
