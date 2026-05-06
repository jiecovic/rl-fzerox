# src/rl_fzerox/core/envs/engine/controls/lean.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import ControllerState
from rl_fzerox.core.domain.lean import (
    DEFAULT_LEAN_MODE,
    LeanMode,
)
from rl_fzerox.core.envs.actions import RACE_CONTROL_MASKS
from rl_fzerox.core.envs.observations import OBSERVATION_STATE_DEFAULTS


@dataclass(slots=True)
class LeanControlState:
    """Track F-Zero X lean button semantics across env steps."""

    mode: LeanMode = DEFAULT_LEAN_MODE
    initial_lockout_frames: int = 0
    left_held: bool = False
    right_held: bool = False
    _lock_index: int = 0
    _lock_remaining_frames: int = 0
    _cooldown_remaining_frames: int = 0

    def reset(self) -> None:
        """Clear held-button and semantic-lock state."""

        self.left_held = False
        self.right_held = False
        self._lock_index = 0
        self._lock_remaining_frames = 0
        self._cooldown_remaining_frames = 0

    def apply_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply the configured lean primitive semantics to one requested action."""

        if self.mode == "minimum_hold":
            return self._apply_minimum_hold(control_state)
        if self.mode == "release_cooldown":
            return self._apply_release_cooldown(control_state)
        return control_state

    def record(self, *, joypad_mask: int, frames_elapsed: int) -> None:
        """Advance held-button and semantic-lock state."""

        left_held = bool(joypad_mask & RACE_CONTROL_MASKS.lean_left)
        right_held = bool(joypad_mask & RACE_CONTROL_MASKS.lean_right)
        previous_lean_index = held_lean_index(
            left_held=self.left_held,
            right_held=self.right_held,
        )
        current_lean_index = lean_index_from_mask(joypad_mask)
        self._update_semantics(
            previous_lean_index=previous_lean_index,
            current_lean_index=current_lean_index,
            frames_elapsed=frames_elapsed,
        )
        self.left_held = left_held
        self.right_held = right_held

    def action_mask_override(self, *, episode_frame_index: int) -> tuple[int, ...] | None:
        """Return live lean branch restrictions implied by the selected mode."""

        if episode_frame_index < max(int(self.initial_lockout_frames), 0):
            return (0,)

        if self.mode == "minimum_hold":
            if self._lock_remaining_frames <= 0 or self._lock_index == 0:
                return None
            return (self._lock_index,)

        if self.mode != "release_cooldown":
            return None
        if self._cooldown_remaining_frames > 0:
            return (0,)

        lean_index = held_lean_index(left_held=self.left_held, right_held=self.right_held)
        if lean_index == 0:
            return None
        return (0, lean_index)

    def _apply_minimum_hold(self, control_state: ControllerState) -> ControllerState:
        if self._lock_remaining_frames <= 0 or self._lock_index == 0:
            return control_state

        if lean_index_from_mask(control_state.joypad_mask) == self._lock_index:
            return control_state
        return replace_lean_index(control_state, self._lock_index)

    def _apply_release_cooldown(self, control_state: ControllerState) -> ControllerState:
        requested_lean_index = lean_index_from_mask(control_state.joypad_mask)
        if requested_lean_index == 0:
            return control_state
        if self._cooldown_remaining_frames > 0:
            return replace_lean_index(control_state, 0)

        current_lean_index = held_lean_index(
            left_held=self.left_held,
            right_held=self.right_held,
        )
        if current_lean_index != 0 and requested_lean_index != current_lean_index:
            return replace_lean_index(control_state, 0)
        return control_state

    def _update_semantics(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if self.mode == "minimum_hold":
            self._update_minimum_hold(
                previous_lean_index=previous_lean_index,
                current_lean_index=current_lean_index,
                frames_elapsed=frames_elapsed,
            )
            return
        if self.mode == "release_cooldown":
            self._update_release_cooldown(
                previous_lean_index=previous_lean_index,
                current_lean_index=current_lean_index,
                frames_elapsed=frames_elapsed,
            )
            return
        self._lock_index = 0
        self._lock_remaining_frames = 0
        self._cooldown_remaining_frames = 0

    def _update_minimum_hold(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if current_lean_index != 0 and current_lean_index != previous_lean_index:
            self._lock_index = current_lean_index
            self._lock_remaining_frames = max(
                OBSERVATION_STATE_DEFAULTS.lean_tap_guard_frames - frames_elapsed,
                0,
            )
            return

        self._lock_remaining_frames = max(self._lock_remaining_frames - frames_elapsed, 0)
        if current_lean_index == 0 and self._lock_remaining_frames == 0:
            self._lock_index = 0

    def _update_release_cooldown(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if previous_lean_index != 0 and current_lean_index != previous_lean_index:
            self._cooldown_remaining_frames = max(
                OBSERVATION_STATE_DEFAULTS.lean_tap_guard_frames - frames_elapsed,
                0,
            )
            return
        if current_lean_index != 0:
            self._cooldown_remaining_frames = 0
            return
        self._cooldown_remaining_frames = max(
            self._cooldown_remaining_frames - frames_elapsed,
            0,
        )


def held_lean_index(*, left_held: bool, right_held: bool) -> int:
    if left_held:
        return 1
    if right_held:
        return 2
    return 0


def lean_index_from_mask(joypad_mask: int) -> int:
    if joypad_mask & RACE_CONTROL_MASKS.lean_left:
        return 1
    if joypad_mask & RACE_CONTROL_MASKS.lean_right:
        return 2
    return 0


def signed_lean(lean_index: int) -> float:
    if lean_index == 1:
        return -1.0
    if lean_index == 2:
        return 1.0
    return 0.0


def replace_lean_index(control_state: ControllerState, lean_index: int) -> ControllerState:
    joypad_mask = control_state.joypad_mask & ~(
        RACE_CONTROL_MASKS.lean_left | RACE_CONTROL_MASKS.lean_right
    )
    if lean_index == 1:
        joypad_mask |= RACE_CONTROL_MASKS.lean_left
    elif lean_index == 2:
        joypad_mask |= RACE_CONTROL_MASKS.lean_right
    return ControllerState(
        joypad_mask=joypad_mask,
        left_stick_x=control_state.left_stick_x,
        left_stick_y=control_state.left_stick_y,
        right_stick_x=control_state.right_stick_x,
        right_stick_y=control_state.right_stick_y,
    )
