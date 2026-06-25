# src/rl_fzerox/core/envs/engine/controls/gates.py
"""Small immutable transforms for controller-state gating.

Higher-level mask and semantics modules use these helpers to clear or preserve
specific control branches while reusing `RaceControlState` replacement code.
"""

from __future__ import annotations

from dataclasses import replace

from fzerox_emulator import RaceControlState


def without_controls(
    control_state: RaceControlState,
    *,
    gas: bool = False,
    air_brake: bool = False,
    boost: bool = False,
    lean_left: bool = False,
    lean_right: bool = False,
) -> RaceControlState:
    """Return a copy with selected semantic buttons cleared."""

    updates: dict[str, bool] = {}
    if gas and control_state.gas:
        updates["gas"] = False
    if air_brake and control_state.air_brake:
        updates["air_brake"] = False
    if boost and control_state.boost:
        updates["boost"] = False
    if lean_left and control_state.lean_left:
        updates["lean_left"] = False
    if lean_right and control_state.lean_right:
        updates["lean_right"] = False
    if not updates:
        return control_state
    return replace(control_state, **updates)


def with_pitch(control_state: RaceControlState, pitch: float) -> RaceControlState:
    if control_state.pitch == pitch:
        return control_state
    return replace(control_state, pitch=pitch)
