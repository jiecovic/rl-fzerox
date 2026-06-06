# src/rl_fzerox/core/envs/engine/controls/semantics.py
"""Shared semantic control filtering for policy and env stepping."""

from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, RaceControlState, SpinRequest
from rl_fzerox.core.runtime_spec.schema.actions import ActionRuntimeConfig

from .availability import apply_dynamic_control_gates
from .history import ControlStateTracker
from .mask_queries import action_branch_value_allowed
from .masks import ActionMaskController


def apply_control_semantics(
    control_state: RaceControlState,
    *,
    mask_controller: ActionMaskController,
    action_config: ActionRuntimeConfig,
    control_state_tracker: ControlStateTracker,
    last_telemetry: FZeroXTelemetry | None,
) -> RaceControlState:
    """Apply live masks and configured lean semantics to one control request."""

    gated_control_state = apply_dynamic_control_gates(
        control_state,
        mask_controller=mask_controller,
        mask_air_brake_on_ground=action_config.mask_air_brake_on_ground,
        continuous_air_brake_mode=action_config.continuous_air_brake_mode,
        last_telemetry=last_telemetry,
    )
    return control_state_tracker.apply_lean_semantics(gated_control_state)


def apply_spin_semantics(
    spin_request: SpinRequest,
    *,
    mask_controller: ActionMaskController,
) -> SpinRequest:
    """Suppress spin requests currently masked by runtime gates."""

    if spin_request == "none":
        return "none"
    spin_index = 1 if spin_request == "left" else 2
    if action_branch_value_allowed(
        mask_controller.action_mask_branches(),
        "spin",
        spin_index,
        missing_allowed=True,
    ):
        return spin_request
    return "none"
