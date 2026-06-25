# src/rl_fzerox/core/envs/engine/controls/air_brake.py
"""Air-brake pulse and hold semantics for requested controller state.

This module isolates timing rules for air-brake inputs before the engine writes
the final controller state to the emulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from fzerox_emulator import RaceControlState


@dataclass(slots=True)
class AirBrakePulseState:
    """Track fixed-length air-brake button pulses across env steps."""

    pulse_frames: int = 0
    _pulse_remaining_frames: int = 0

    def reset(self) -> None:
        """Clear any active air-brake pulse."""

        self._pulse_remaining_frames = 0

    def apply_semantics(
        self,
        control_state: RaceControlState,
        *,
        available: bool,
    ) -> RaceControlState:
        """Apply the active pulse to one requested action."""

        if self.pulse_frames <= 0:
            return control_state
        if not available:
            return without_air_brake(control_state)
        if self._pulse_remaining_frames > 0:
            return with_air_brake(control_state)
        return control_state

    def record(
        self,
        *,
        control_state: RaceControlState,
        requested_control_state: RaceControlState,
        frames_elapsed: int,
    ) -> None:
        """Advance pulse state by the native frames consumed by one env step."""

        pulse_frames = max(int(self.pulse_frames), 0)
        elapsed = max(int(frames_elapsed), 0)
        if pulse_frames <= 0:
            self._pulse_remaining_frames = 0
            return

        if self._pulse_remaining_frames > 0:
            self._pulse_remaining_frames = max(self._pulse_remaining_frames - elapsed, 0)
            return

        if requested_control_state.air_brake and control_state.air_brake:
            self._pulse_remaining_frames = max(pulse_frames - elapsed, 0)

    def action_mask_override(self) -> tuple[int, ...] | None:
        """Return live air-brake branch restrictions implied by an active pulse."""

        if self._pulse_remaining_frames <= 0:
            return None
        return (0,)

    @property
    def pulse_active(self) -> bool:
        return self._pulse_remaining_frames > 0

    @property
    def pulse_remaining_frames(self) -> int:
        return self._pulse_remaining_frames


def with_air_brake(control_state: RaceControlState) -> RaceControlState:
    return replace(control_state, air_brake=True)


def without_air_brake(control_state: RaceControlState) -> RaceControlState:
    return replace(control_state, air_brake=False)
