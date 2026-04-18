# src/rl_fzerox/core/envs/engine/control_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fzerox_emulator import ControllerState
from rl_fzerox.core.domain.lean import (
    DEFAULT_LEAN_MODE,
    LEAN_MODE_MINIMUM_HOLD,
    LEAN_MODE_RELEASE_COOLDOWN,
    LeanMode,
)
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.core.envs.observations import (
    DEFAULT_ACTION_HISTORY_CONTROLS,
    DEFAULT_ACTION_HISTORY_LEN,
    LEAN_DOUBLE_TAP_WINDOW_FRAMES,
    RECENT_BOOST_PRESSURE_WINDOW_FRAMES,
    RECENT_STEER_PRESSURE_WINDOW_FRAMES,
    ActionHistoryControl,
)


@dataclass(frozen=True, slots=True)
class _ActionHistorySample:
    steer: float
    gas: float
    air_brake: float
    boost: float
    lean: float


@dataclass(slots=True)
class ControlStateTracker:
    """Track recent action-derived state that the policy also observes.

    Telemetry tells us what the game is doing now. This tracker complements it
    with short-lived input history for mechanics whose behavior depends on
    recent button edges, such as F-Zero X Z/R lean taps.
    """

    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    boost_decision_interval_frames: int = 1
    boost_request_lockout_frames: int = 0
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS
    _recent_boost_frames: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_BOOST_PRESSURE_WINDOW_FRAMES),
    )
    _recent_steer_frames: deque[float] = field(
        default_factory=lambda: deque(maxlen=RECENT_STEER_PRESSURE_WINDOW_FRAMES),
    )
    _action_history: deque[_ActionHistorySample] = field(init=False)
    _resolved_action_history_len: int = field(init=False, default=0)
    _recent_boost_frame_sum: int = 0
    _recent_steer_frame_sum: float = 0.0
    _left_lean_held: bool = False
    _right_lean_held: bool = False
    _left_steer_held: bool = False
    _right_steer_held: bool = False
    _left_press_age_frames: int = LEAN_DOUBLE_TAP_WINDOW_FRAMES
    _right_press_age_frames: int = LEAN_DOUBLE_TAP_WINDOW_FRAMES
    _lean_lock_index: int = 0
    _lean_lock_remaining_frames: int = 0
    _lean_cooldown_remaining_frames: int = 0
    _episode_frame_index: int = 0
    _next_boost_decision_frame: int = 0
    _boost_request_lockout_remaining_frames: int = 0

    def __post_init__(self) -> None:
        self._resolved_action_history_len = _resolve_action_history_len(self.action_history_len)
        self._action_history = deque(maxlen=self._resolved_action_history_len)

    def reset(self) -> None:
        """Clear all step-to-step control history."""

        self._recent_boost_frames.clear()
        self._recent_steer_frames.clear()
        self._action_history.clear()
        self._recent_boost_frame_sum = 0
        self._recent_steer_frame_sum = 0.0
        self._left_lean_held = False
        self._right_lean_held = False
        self._left_steer_held = False
        self._right_steer_held = False
        self._left_press_age_frames = LEAN_DOUBLE_TAP_WINDOW_FRAMES
        self._right_press_age_frames = LEAN_DOUBLE_TAP_WINDOW_FRAMES
        self._lean_lock_index = 0
        self._lean_lock_remaining_frames = 0
        self._lean_cooldown_remaining_frames = 0
        self._episode_frame_index = 0
        self._next_boost_decision_frame = 0
        self._boost_request_lockout_remaining_frames = 0

    def apply_lean_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply the selected Z/R lean primitive semantics to one action."""

        if self.lean_mode == LEAN_MODE_MINIMUM_HOLD:
            return self._apply_minimum_hold(control_state)
        if self.lean_mode == LEAN_MODE_RELEASE_COOLDOWN:
            return self._apply_release_cooldown(control_state)
        return control_state

    def record_step(
        self,
        *,
        control_state: ControllerState,
        frames_run: int,
        gas_level: float | None = None,
    ) -> None:
        """Advance tracked control history by one env step."""

        frames_elapsed = max(int(frames_run), 0)
        boost_requested = bool(control_state.joypad_mask & BOOST_MASK)
        steer_axis = _clamp(float(control_state.left_stick_x), -1.0, 1.0)
        left_held = bool(control_state.joypad_mask & LEAN_LEFT_MASK)
        right_held = bool(control_state.joypad_mask & LEAN_RIGHT_MASK)
        previous_lean_index = _held_lean_index(
            left_held=self._left_lean_held,
            right_held=self._right_lean_held,
        )
        current_lean_index = _lean_index_from_mask(control_state.joypad_mask)

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
            was_held=self._left_lean_held,
            is_held=left_held,
            frames_elapsed=frames_elapsed,
        )
        self._right_press_age_frames = _advance_press_age(
            self._right_press_age_frames,
            was_held=self._right_lean_held,
            is_held=right_held,
            frames_elapsed=frames_elapsed,
        )
        self._update_lean_semantics(
            previous_lean_index=previous_lean_index,
            current_lean_index=current_lean_index,
            frames_elapsed=frames_elapsed,
        )
        self._left_lean_held = left_held
        self._right_lean_held = right_held
        self._left_steer_held = steer_axis < -1.0e-6
        self._right_steer_held = steer_axis > 1.0e-6
        self._record_action_history(control_state, gas_level=gas_level)

    def observation_fields(self) -> dict[str, float]:
        """Return control-history features passed into observation building."""

        return {
            "left_lean_held": float(self._left_lean_held),
            "right_lean_held": float(self._right_lean_held),
            # Normalize against the game's 15-frame double-tap window so the
            # policy can distinguish a fresh tap from an old one.
            "left_press_age_norm": _lean_press_age_norm(self._left_press_age_frames),
            "right_press_age_norm": _lean_press_age_norm(self._right_press_age_frames),
            "recent_boost_pressure": self._recent_boost_pressure(),
            "steer_left_held": float(self._left_steer_held),
            "steer_right_held": float(self._right_steer_held),
            "recent_steer_pressure": self._recent_steer_pressure(),
        }

    def action_history_fields(self) -> dict[str, float]:
        """Return fixed-width previous-action features for policy observations."""

        return self._action_history_fields()

    def boost_action_allowed_by_timing(self) -> bool:
        """Return whether this frame is a tactical manual-boost decision slot."""

        if self._boost_request_lockout_remaining_frames > 0:
            return False
        interval = max(int(self.boost_decision_interval_frames), 1)
        return interval <= 1 or self._episode_frame_index >= self._next_boost_decision_frame

    def lean_action_mask_override(self) -> tuple[int, ...] | None:
        """Return live lean branch restrictions implied by the selected mode."""

        if self.lean_mode == LEAN_MODE_MINIMUM_HOLD:
            if self._lean_lock_remaining_frames <= 0 or self._lean_lock_index == 0:
                return None
            return (self._lean_lock_index,)

        if self.lean_mode != LEAN_MODE_RELEASE_COOLDOWN:
            return None
        if self._lean_cooldown_remaining_frames > 0:
            return (0,)

        lean_index = _held_lean_index(
            left_held=self._left_lean_held,
            right_held=self._right_lean_held,
        )
        if lean_index == 0:
            return None
        return (0, lean_index)

    def _apply_minimum_hold(self, control_state: ControllerState) -> ControllerState:
        if self._lean_lock_remaining_frames <= 0 or self._lean_lock_index == 0:
            return control_state

        if _lean_index_from_mask(control_state.joypad_mask) == self._lean_lock_index:
            return control_state
        return _replace_lean_index(control_state, self._lean_lock_index)

    def _apply_release_cooldown(self, control_state: ControllerState) -> ControllerState:
        requested_lean_index = _lean_index_from_mask(control_state.joypad_mask)
        if requested_lean_index == 0:
            return control_state
        if self._lean_cooldown_remaining_frames > 0:
            return _replace_lean_index(control_state, 0)

        current_lean_index = _held_lean_index(
            left_held=self._left_lean_held,
            right_held=self._right_lean_held,
        )
        if current_lean_index != 0 and requested_lean_index != current_lean_index:
            return _replace_lean_index(control_state, 0)
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

    def _record_action_history(
        self,
        control_state: ControllerState,
        *,
        gas_level: float | None,
    ) -> None:
        joypad = control_state.joypad_mask
        normalized_gas = (
            (1.0 if joypad & ACCELERATE_MASK else 0.0)
            if gas_level is None
            else _clamp(float(gas_level), 0.0, 1.0)
        )
        self._action_history.append(
            _ActionHistorySample(
                steer=_clamp(float(control_state.left_stick_x), -1.0, 1.0),
                gas=normalized_gas,
                air_brake=1.0 if joypad & AIR_BRAKE_MASK else 0.0,
                boost=1.0 if joypad & BOOST_MASK else 0.0,
                lean=_signed_lean(_lean_index_from_mask(joypad)),
            )
        )

    def _action_history_fields(self) -> dict[str, float]:
        samples = list(reversed(self._action_history))
        fields: dict[str, float] = {}
        for index in range(self._resolved_action_history_len):
            sample = samples[index] if index < len(samples) else _empty_action_history_sample()
            suffix = index + 1
            fields[f"prev_steer_{suffix}"] = sample.steer
            fields[f"prev_gas_{suffix}"] = sample.gas
            fields[f"prev_air_brake_{suffix}"] = sample.air_brake
            fields[f"prev_boost_{suffix}"] = sample.boost
            fields[f"prev_lean_{suffix}"] = sample.lean
        return {
            key: value
            for key, value in fields.items()
            if _action_history_field_control(key) in self.action_history_controls
        }

    def _update_boost_timing(self, *, boost_requested: bool, frames_elapsed: int) -> None:
        lockout_frames = max(int(self.boost_request_lockout_frames), 0)
        boost_decision_slot_was_open = self._episode_frame_index >= self._next_boost_decision_frame
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

    def _update_lean_semantics(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if self.lean_mode == LEAN_MODE_MINIMUM_HOLD:
            self._update_minimum_hold(
                previous_lean_index=previous_lean_index,
                current_lean_index=current_lean_index,
                frames_elapsed=frames_elapsed,
            )
            return
        if self.lean_mode == LEAN_MODE_RELEASE_COOLDOWN:
            self._update_release_cooldown(
                previous_lean_index=previous_lean_index,
                current_lean_index=current_lean_index,
                frames_elapsed=frames_elapsed,
            )
            return
        self._lean_lock_index = 0
        self._lean_lock_remaining_frames = 0
        self._lean_cooldown_remaining_frames = 0

    def _update_minimum_hold(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if current_lean_index != 0 and current_lean_index != previous_lean_index:
            self._lean_lock_index = current_lean_index
            self._lean_lock_remaining_frames = max(
                LEAN_DOUBLE_TAP_WINDOW_FRAMES - frames_elapsed,
                0,
            )
            return

        self._lean_lock_remaining_frames = max(
            self._lean_lock_remaining_frames - frames_elapsed,
            0,
        )
        if current_lean_index == 0 and self._lean_lock_remaining_frames == 0:
            self._lean_lock_index = 0

    def _update_release_cooldown(
        self,
        *,
        previous_lean_index: int,
        current_lean_index: int,
        frames_elapsed: int,
    ) -> None:
        if previous_lean_index != 0 and current_lean_index != previous_lean_index:
            self._lean_cooldown_remaining_frames = max(
                LEAN_DOUBLE_TAP_WINDOW_FRAMES - frames_elapsed,
                0,
            )
            return
        if current_lean_index != 0:
            self._lean_cooldown_remaining_frames = 0
            return
        self._lean_cooldown_remaining_frames = max(
            self._lean_cooldown_remaining_frames - frames_elapsed,
            0,
        )


def _held_lean_index(*, left_held: bool, right_held: bool) -> int:
    if left_held:
        return 1
    if right_held:
        return 2
    return 0


def _lean_index_from_mask(joypad_mask: int) -> int:
    if joypad_mask & LEAN_LEFT_MASK:
        return 1
    if joypad_mask & LEAN_RIGHT_MASK:
        return 2
    return 0


def _signed_lean(lean_index: int) -> float:
    if lean_index == 1:
        return -1.0
    if lean_index == 2:
        return 1.0
    return 0.0


def _empty_action_history_sample() -> _ActionHistorySample:
    return _ActionHistorySample(
        steer=0.0,
        gas=0.0,
        air_brake=0.0,
        boost=0.0,
        lean=0.0,
    )


def _resolve_action_history_len(action_history_len: int | None) -> int:
    if action_history_len is None:
        return 0
    length = int(action_history_len)
    if length <= 0:
        raise ValueError("action_history_len must be positive or None")
    return length


def _action_history_field_control(field_name: str) -> ActionHistoryControl:
    control_name = field_name.removeprefix("prev_").rsplit("_", maxsplit=1)[0]
    if control_name == "steer":
        return "steer"
    if control_name == "gas":
        return "gas"
    if control_name == "air_brake":
        return "air_brake"
    if control_name == "boost":
        return "boost"
    if control_name == "lean":
        return "lean"
    raise ValueError(f"Unsupported action history field: {field_name!r}")


def _replace_lean_index(control_state: ControllerState, lean_index: int) -> ControllerState:
    joypad_mask = control_state.joypad_mask & ~(LEAN_LEFT_MASK | LEAN_RIGHT_MASK)
    if lean_index == 1:
        joypad_mask |= LEAN_LEFT_MASK
    elif lean_index == 2:
        joypad_mask |= LEAN_RIGHT_MASK
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
        return min(frames_elapsed, LEAN_DOUBLE_TAP_WINDOW_FRAMES)
    return min(previous_age_frames + frames_elapsed, LEAN_DOUBLE_TAP_WINDOW_FRAMES)


def _lean_press_age_norm(frames: int) -> float:
    clamped_frames = min(max(int(frames), 0), LEAN_DOUBLE_TAP_WINDOW_FRAMES)
    return clamped_frames / LEAN_DOUBLE_TAP_WINDOW_FRAMES


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
