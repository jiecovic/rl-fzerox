# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.domain.action_adapters import (
    ACTION_ADAPTERS,
    DEFAULT_ACTION_ADAPTER_NAME,
    ActionAdapterName,
)
from rl_fzerox.core.envs.actions.base import (
    ActionAdapter,
    ActionValue,
    DiscreteActionDimension,
    ResettableActionAdapter,
)
from rl_fzerox.core.envs.actions.buttons import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    BRAKE_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    THROTTLE_MASK,
)
from rl_fzerox.core.envs.actions.configured import (
    ConfiguredDiscreteActionAdapter,
    ConfiguredHybridActionAdapter,
)

ActionAdapterFactory = Callable[[ActionRuntimeConfig], ActionAdapter]
DEFAULT_ACTION_NAME: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
ACTION_ADAPTER_REGISTRY: dict[ActionAdapterName, ActionAdapterFactory] = {
    ACTION_ADAPTERS.configured_discrete: ConfiguredDiscreteActionAdapter,
    ACTION_ADAPTERS.configured_hybrid: ConfiguredHybridActionAdapter,
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
    "DEFAULT_ACTION_NAME",
    "DiscreteActionDimension",
    "LEAN_LEFT_MASK",
    "LEAN_RIGHT_MASK",
    "ResettableActionAdapter",
    "THROTTLE_MASK",
    "action_adapter_names",
    "build_action_adapter",
]
