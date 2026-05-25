# src/rl_fzerox/core/envs/actions/__init__.py
from collections.abc import Callable

from rl_fzerox.core.domain.action_adapters import ActionAdapterName
from rl_fzerox.core.envs.actions.base import (
    ActionAdapter,
    ActionValue,
    DecodedAction,
    DiscreteActionDimension,
    ResettableActionAdapter,
)
from rl_fzerox.core.envs.actions.buttons import RACE_CONTROL_MASKS
from rl_fzerox.core.envs.actions.configured import (
    ConfiguredDiscreteActionAdapter,
    ConfiguredHybridActionAdapter,
)
from rl_fzerox.core.runtime_spec.schema import ActionConfig, ActionRuntimeConfig

ActionAdapterFactory = Callable[[ActionRuntimeConfig], ActionAdapter]
ACTION_ADAPTER_REGISTRY: dict[ActionAdapterName, ActionAdapterFactory] = {
    "configured_discrete": ConfiguredDiscreteActionAdapter,
    "configured_hybrid": ConfiguredHybridActionAdapter,
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
    "ActionAdapter",
    "ActionAdapterFactory",
    "ActionValue",
    "ConfiguredDiscreteActionAdapter",
    "ConfiguredHybridActionAdapter",
    "DecodedAction",
    "DiscreteActionDimension",
    "RACE_CONTROL_MASKS",
    "ResettableActionAdapter",
    "action_adapter_names",
    "build_action_adapter",
]
