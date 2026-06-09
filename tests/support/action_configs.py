# tests/support/action_configs.py
from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import ActionConfig
from rl_fzerox.core.runtime_spec.schema.actions import (
    ConfiguredContinuousAxis,
    ConfiguredDiscreteAxis,
)


def configured_discrete_action(
    *layout_discrete_axes: ConfiguredDiscreteAxis,
    **overrides: object,
) -> ActionConfig:
    return ActionConfig.model_validate(
        {
            "adapter_name": "configured_discrete",
            "layout_discrete_axes": layout_discrete_axes,
            **overrides,
        }
    )


def configured_hybrid_action(
    *,
    continuous_axes: tuple[ConfiguredContinuousAxis, ...],
    discrete_axes: tuple[ConfiguredDiscreteAxis, ...] = (),
    **overrides: object,
) -> ActionConfig:
    return ActionConfig.model_validate(
        {
            "adapter_name": "configured_hybrid",
            "layout_continuous_axes": continuous_axes,
            "layout_discrete_axes": discrete_axes,
            **overrides,
        }
    )
