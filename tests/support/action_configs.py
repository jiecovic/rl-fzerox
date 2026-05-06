# tests/support/action_configs.py
from __future__ import annotations

from rl_fzerox.core.config.schema import ActionConfig


def configured_discrete_action(
    *layout_discrete_axes: str,
    **overrides: object,
) -> ActionConfig:
    return ActionConfig(
        name="configured_discrete",
        layout_discrete_axes=layout_discrete_axes,
        **overrides,
    )


def configured_hybrid_action(
    *,
    continuous_axes: tuple[str, ...],
    discrete_axes: tuple[str, ...] = (),
    **overrides: object,
) -> ActionConfig:
    return ActionConfig(
        name="configured_hybrid",
        layout_continuous_axes=continuous_axes,
        layout_discrete_axes=discrete_axes,
        **overrides,
    )
