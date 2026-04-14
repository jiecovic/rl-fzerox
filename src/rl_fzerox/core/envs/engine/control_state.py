# src/rl_fzerox/core/envs/engine/control_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fzerox_emulator import ControllerState
from rl_fzerox.core.domain.shoulder_slide import (
    DEFAULT_SHOULDER_SLIDE_MODE,
    SHOULDER_SLIDE_MODE_MINIMUM_HOLD,
    SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN,
    ShoulderSlideMode,
)
from rl_fzerox.core.envs.actions import BOOST_MASK, SHOULDER_LEFT_MASK, SHOULDER_RIGHT_MASK
from rl_fzerox.core.envs.observations import (
    RECENT_BOOST_PRESSURE_WINDOW_FRAMES,
    RECENT_STEER_PRESSURE_WINDOW_FRAMES,
    SHOULDER_DOUBLE_TAP_WINDOW_FRAMES,
)


@dataclass(slots=True)
class ControlStateTracker:
    """Track recent action-derived state that the policy also observes.

    Telemetry tells us what the game is doing now. This tracker complements it
    with short-lived input history for mechanics whose behavior depends on
    recent button edges, such as F-Zero X shoulder taps.
    """

    shoulder_slide_mode: ShoulderSlideMode = DEFAULT_SHOULDER_SLIDE_MODE
    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 0
    _recent_boost_frames: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_BOOST_PRESSURE_WINDOW_FRAMES),
    )
    _recent_steer_frames: deque[float] = field(
        default_factory=lambda: deque(maxlen=RECENT_STEER_PRESSURE_WINDOW_FRAMES),
    )
    _recent_boost_frame_sum: int = 0
    _recent_steer_frame_sum: float = 0.0
    _left_shoulder_held: bool = False
    _right_shoulder_held: bool = False
    _left_steer_held: bool = False
    _right_steer_held: bool = False
    _left_press_age_frames: int = SHOULDER_DOUBLE_TAP_WINDOW_FRAMES
    _right_press_age_frames: int = SHOULDER_DOUBLE_TAP_WINDOW_FRAMES
    _shoulder_lock_index: int = 0
    _shoulder_lock_remaining_frames: int = 0
    _shoulder_cooldown_remaining_frames: int = 0
    _episode_frame_index: int = 0
    _next_boost_decision_frame: int = 0
    _boost_request_lockout_remaining_frames: int = 0

    def reset(self) -> None:
        """Clear all step-to-step control history."""

        self._recent_boost_frames.clear()
        self._recent_steer_frames.clear()
        self._recent_boost_frame_sum = 0
        self._recent_steer_frame_sum = 0.0
        self._left_shoulder_held = False
        self._right_shoulder_held = False
        self._left_steer_held = False
        self._right_steer_held = False
        self._left_press_age_frames = SHOULDER_DOUBLE_TAP_WINDOW_FRAMES
        self._right_press_age_frames = SHOULDER_DOUBLE_TAP_WINDOW_FRAMES
        self._shoulder_lock_index = 0
        self._shoulder_lock_remaining_frames = 0
        self._shoulder_cooldown_remaining_frames = 0
        self._episode_frame_index = 0
        self._next_boost_decision_frame = 0
        self._boost_request_lockout_remaining_frames = 0

    def apply_shoulder_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply the selected shoulder-slide primitive semantics to one action."""

        if self.shoulder_slide_mode == SHOULDER_SLIDE_MODE_MINIMUM_HOLD:
            return self._apply_minimum_hold(control_state)
        if self.shoulder_slide_mode == SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN:
            return self._apply_release_cooldown(control_state)
        return control_state

    def record_step(self, *, control_state: ControllerState, frames_run: int) -> None:
        """Advance tracked control history by one env step."""

        frames_elapsed = max(int(frames_run), 0)
        boost_requested = bool(control_state.joypad_mask & BOOST_MASK)
        steer_axis = _clamp(float(control_state.left_stick_x), -1.0, 1.0)
        left_held = bool(control_state.joypad_mask & SHOULDER_LEFT_MASK)
        right_held = bool(control_state.joypad_mask & SHOULDER_RIGHT_MASK)
        previous_shoulder_index = _held_shoulder_index(
            left_held=self._left_shoulder_held,
            right_held=self._right_shoulder_held,
        )
        current_shoulder_index = _shoulder_index_from_mask(control_state.joypad_mask)

        self._record_recent_boost_pressure(
            boost_requested=boost_requested,
            frames_run=frames_elapsed,
        )
        self._record_recent_steer_pressure(
            steer_axis=steer_axis,
            frames_run=frames_elapsed,
        )
        self._update_boost_timing(
            boost_requested=boost_requested,
            frames_elapsed=frames_elapsed,
        )
        self._left_press_age_frames = _advance_press_age(
            self._left_press_age_frames,
            was_held=self._left_shoulder_held,
            is_held=left_held,
            frames_elapsed=frames_elapsed,
        )
        self._right_press_age_frames = _advance_press_age(
            self._right_press_age_frames,
            was_held=self._right_shoulder_held,
            is_held=right_held,
            frames_elapsed=frames_elapsed,
        )
        self._update_shoulder_semantics(
            previous_shoulder_index=previous_shoulder_index,
            current_shoulder_index=current_shoulder_index,
            frames_elapsed=frames_elapsed,
        )
        self._left_shoulder_held = left_held
        self._right_shoulder_held = right_held
        self._left_steer_held = steer_axis < -1.0e-6
        self._right_steer_held = steer_axis > 1.0e-6

    def observation_fields(self) -> dict[str, float]:
        """Return control-history features passed into observation building."""

        return {
            "left_shoulder_held": float(self._left_shoulder_held),
            "right_shoulder_held": float(self._right_shoulder_held),
            # Normalize against the game's 15-frame double-tap window so the
            # policy can distinguish a fresh tap from an old one.
            "left_press_age_norm": _shoulder_press_age_norm(self._left_press_age_frames),
            "right_press_age_norm": _shoulder_press_age_norm(self._right_press_age_frames),
            "recent_boost_pressure": self._recent_boost_pressure(),
            "steer_left_held": float(self._left_steer_held),
            "steer_right_held": float(self._right_steer_held),
            "recent_steer_pressure": self._recent_steer_pressure(),
        }

    def boost_action_allowed_by_timing(self) -> bool:
        """Return whether this frame is a tactical manual-boost decision slot."""

        if self._boost_request_lockout_remaining_frames > 0:
            return False
        interval = max(int(self.boost_decision_interval_frames), 1)
        return interval <= 1 or self._episode_frame_index >= self._next_boost_decision_frame

    def shoulder_action_mask_override(self) -> tuple[int, ...] | None:
        """Return live shoulder branch restrictions implied by the selected mode."""

        if self.shoulder_slide_mode == SHOULDER_SLIDE_MODE_MINIMUM_HOLD:
            if self._shoulder_lock_remaining_frames <= 0 or self._shoulder_lock_index == 0:
                return None
            return (self._shoulder_lock_index,)

        if self.shoulder_slide_mode != SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN:
            return None
        if self._shoulder_cooldown_remaining_frames > 0:
            return (0,)

        shoulder_index = _held_shoulder_index(
            left_held=self._left_shoulder_held,
            right_held=self._right_shoulder_held,
        )
        if shoulder_index == 0:
            return None
        return (0, shoulder_index)

    def _apply_minimum_hold(self, control_state: ControllerState) -> ControllerState:
        if self._shoulder_lock_remaining_frames <= 0 or self._shoulder_lock_index == 0:
            return control_state

        if _shoulder_index_from_mask(control_state.joypad_mask) == self._shoulder_lock_index:
            return control_state
        return _replace_shoulder_index(control_state, self._shoulder_lock_index)

    def _apply_release_cooldown(self, control_state: ControllerState) -> ControllerState:
        requested_shoulder_index = _shoulder_index_from_mask(control_state.joypad_mask)
        if requested_shoulder_index == 0:
            return control_state
        if self._shoulder_cooldown_remaining_frames > 0:
            return _replace_shoulder_index(control_state, 0)

        current_shoulder_index = _held_shoulder_index(
            left_held=self._left_shoulder_held,
            right_held=self._right_shoulder_held,
        )
        if current_shoulder_index != 0 and requested_shoulder_index != current_shoulder_index:
            return _replace_shoulder_index(control_state, 0)
        return control_state

    def _record_recent_boost_pressure(self, *, boost_requested: bool, frames_run: int) -> None:
        encoded = 1 if boost_requested else 0
        for _ in range(frames_run):
            if len(self._recent_boost_frames) == self._recent_boost_frames.maxlen:
                removed = self._recent_boost_frames.popleft()
                self._recent_boost_frame_sum -= removed
            self._recent_boost_frames.append(encoded)
            self._recent_boost_frame_sum += encoded

    def _recent_boost_pressure(self) -> float:
        if not self._recent_boost_frames:
            return 0.0
        return self._recent_boost_frame_sum / len(self._recent_boost_frames)

    def _record_recent_steer_pressure(self, *, steer_axis: float, frames_run: int) -> None:
        for _ in range(frames_run):
            if len(self._recent_steer_frames) == self._recent_steer_frames.maxlen:
                removed = self._recent_steer_frames.popleft()
                self._recent_steer_frame_sum -= removed
            self._recent_steer_frames.append(steer_axis)
            self._recent_steer_frame_sum += steer_axis

    def _recent_steer_pressure(self) -> float:
        if not self._recent_steer_frames:
            return 0.0
        window_size = self._recent_steer_frames.maxlen
        if window_size is None or window_size <= 0:
            return 0.0
        return self._recent_steer_frame_sum / window_size

    def _update_boost_timing(self, *, boost_requested: bool, frames_elapsed: int) -> None:
        lockout_frames = max(int(self.boost_request_lockout_frames), 0)
        boost_decision_slot_was_open = (
            self._episode_frame_index >= self._next_boost_decision_frame
        )
        if boost_requested and lockout_frames > 0:
            self._boost_request_lockout_remaining_frames = max(
                lockout_frames - frames_elapsed,
                0,
            )
        else:
            self._boost_request_lockout_remaining_frames = max(
                self._boost_request_lockout_remaining_frames - frames_elapsed,
                0,
            )
        if boost_decision_slot_was_open:
            interval = max(int(self.boost_decision_interval_frames), 1)
            while self._next_boost_decision_frame <= self._episode_frame_index:
                self._next_boost_decision_frame += interval
        self._episode_frame_index += frames_elapsed

    def _update_shoulder_semantics(
        self,
        *,
        previous_shoulder_index: int,
        current_shoulder_index: int,
        frames_elapsed: int,
    ) -> None:
        if self.shoulder_slide_mode == SHOULDER_SLIDE_MODE_MINIMUM_HOLD:
            self._update_minimum_hold(
                previous_shoulder_index=previous_shoulder_index,
                current_shoulder_index=current_shoulder_index,
                frames_elapsed=frames_elapsed,
            )
            return
        if self.shoulder_slide_mode == SHOULDER_SLIDE_MODE_RELEASE_COOLDOWN:
            self._update_release_cooldown(
                previous_shoulder_index=previous_shoulder_index,
                current_shoulder_index=current_shoulder_index,
                frames_elapsed=frames_elapsed,
            )
            return
        self._shoulder_lock_index = 0
        self._shoulder_lock_remaining_frames = 0
        self._shoulder_cooldown_remaining_frames = 0

    def _update_minimum_hold(
        self,
        *,
        previous_shoulder_index: int,
        current_shoulder_index: int,
        frames_elapsed: int,
    ) -> None:
        if current_shoulder_index != 0 and current_shoulder_index != previous_shoulder_index:
            self._shoulder_lock_index = current_shoulder_index
            self._shoulder_lock_remaining_frames = max(
                SHOULDER_DOUBLE_TAP_WINDOW_FRAMES - frames_elapsed,
                0,
            )
            return

        self._shoulder_lock_remaining_frames = max(
            self._shoulder_lock_remaining_frames - frames_elapsed,
            0,
        )
        if current_shoulder_index == 0 and self._shoulder_lock_remaining_frames == 0:
            self._shoulder_lock_index = 0

    def _update_release_cooldown(
        self,
        *,
        previous_shoulder_index: int,
        current_shoulder_index: int,
        frames_elapsed: int,
    ) -> None:
        if previous_shoulder_index != 0 and current_shoulder_index != previous_shoulder_index:
            self._shoulder_cooldown_remaining_frames = max(
                SHOULDER_DOUBLE_TAP_WINDOW_FRAMES - frames_elapsed,
                0,
            )
            return
        if current_shoulder_index != 0:
            self._shoulder_cooldown_remaining_frames = 0
            return
        self._shoulder_cooldown_remaining_frames = max(
            self._shoulder_cooldown_remaining_frames - frames_elapsed,
            0,
        )


def _held_shoulder_index(*, left_held: bool, right_held: bool) -> int:
    if left_held:
        return 1
    if right_held:
        return 2
    return 0


def _shoulder_index_from_mask(joypad_mask: int) -> int:
    if joypad_mask & SHOULDER_LEFT_MASK:
        return 1
    if joypad_mask & SHOULDER_RIGHT_MASK:
        return 2
    return 0


def _replace_shoulder_index(control_state: ControllerState, shoulder_index: int) -> ControllerState:
    joypad_mask = control_state.joypad_mask & ~(SHOULDER_LEFT_MASK | SHOULDER_RIGHT_MASK)
    if shoulder_index == 1:
        joypad_mask |= SHOULDER_LEFT_MASK
    elif shoulder_index == 2:
        joypad_mask |= SHOULDER_RIGHT_MASK
    return ControllerState(
        joypad_mask=joypad_mask,
        left_stick_x=control_state.left_stick_x,
        left_stick_y=control_state.left_stick_y,
        right_stick_x=control_state.right_stick_x,
        right_stick_y=control_state.right_stick_y,
    )


def _advance_press_age(
    previous_age_frames: int,
    *,
    was_held: bool,
    is_held: bool,
    frames_elapsed: int,
) -> int:
    if is_held and not was_held:
        return min(frames_elapsed, SHOULDER_DOUBLE_TAP_WINDOW_FRAMES)
    return min(previous_age_frames + frames_elapsed, SHOULDER_DOUBLE_TAP_WINDOW_FRAMES)


def _shoulder_press_age_norm(frames: int) -> float:
    clamped_frames = min(max(int(frames), 0), SHOULDER_DOUBLE_TAP_WINDOW_FRAMES)
    return clamped_frames / SHOULDER_DOUBLE_TAP_WINDOW_FRAMES


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
