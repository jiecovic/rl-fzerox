# src/rl_fzerox/core/envs/actions/hybrid/common.py
from __future__ import annotations

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions.base import shape_steer_value
from rl_fzerox.core.envs.actions.discrete.steer_drive import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive_boost_lean import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.core.envs.actions.hybrid.layouts import PITCH_BUCKETS, HybridActionLayout
from rl_fzerox.core.envs.actions.hybrid.spaces import hybrid_action_mask


def button_mask(
    *,
    gas: int,
    boost: int,
    lean: int,
    air_brake: int = 0,
) -> int:
    """Build joypad button bits from independent discrete action heads."""

    joypad_mask = 0
    if gas == 1:
        joypad_mask |= ACCELERATE_MASK
    if air_brake == 1:
        joypad_mask |= AIR_BRAKE_MASK
    if boost == 1:
        joypad_mask |= BOOST_MASK
    return apply_lean_mask(joypad_mask, lean)


def apply_lean_mask(joypad_mask: int, lean: int) -> int:
    """Apply the shared lean branch mapping to a joypad mask."""

    if lean == 1:
        return joypad_mask | LEAN_LEFT_MASK
    if lean == 2:
        return joypad_mask | LEAN_RIGHT_MASK
    return joypad_mask


def controller_state(
    *,
    joypad_mask: int,
    steer: float,
    response_power: float,
    pitch: float = 0.0,
) -> ControllerState:
    """Build emulator controller state from decoded hybrid action values."""

    return ControllerState(
        joypad_mask=joypad_mask,
        left_stick_x=shape_steer_value(
            steer,
            response_power=response_power,
        ),
        left_stick_y=pitch,
    )


def pitch_value(pitch_index: int) -> float:
    """Map the configured five-bucket pitch head to the N64 stick y-axis."""

    if PITCH_BUCKETS.count != 5 or PITCH_BUCKETS.neutral_index != 2:
        raise RuntimeError("pitch bucket mapping expects five buckets with neutral at index 2")
    return (float(pitch_index) - float(PITCH_BUCKETS.neutral_index)) / 2.0


def action_mask(
    layout: HybridActionLayout,
    *,
    base_overrides: dict[str, tuple[int, ...]] | None = None,
    stage_overrides: dict[str, tuple[int, ...]] | None = None,
    dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
) -> ActionMask:
    """Build one flattened mask for a hybrid action layout."""

    return hybrid_action_mask(
        layout.dimensions,
        base_overrides=base_overrides,
        stage_overrides=stage_overrides,
        dynamic_overrides=dynamic_overrides,
    )
