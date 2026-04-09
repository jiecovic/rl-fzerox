# src/rl_fzerox/core/envs/engine/control_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fzerox_emulator import ControllerState
from rl_fzerox.core.envs.actions import BOOST_MASK, DRIFT_LEFT_MASK, DRIFT_RIGHT_MASK
from rl_fzerox.core.envs.observations import (
    DRIFT_DOUBLE_TAP_WINDOW_FRAMES,
    RECENT_BOOST_PRESSURE_WINDOW_FRAMES,
)


@dataclass(slots=True)
class ControlStateTracker:
    """Track recent action-derived state that the policy also observes.

    Telemetry tells us what the game is doing now. This tracker complements it
    with short-lived input history for mechanics whose behavior depends on
    recent button edges, such as F-Zero X shoulder taps.
    """

    _recent_boost_frames: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_BOOST_PRESSURE_WINDOW_FRAMES),
    )
    _recent_boost_frame_sum: int = 0
    _left_drift_held: bool = False
    _right_drift_held: bool = False
    _left_press_age_frames: int = DRIFT_DOUBLE_TAP_WINDOW_FRAMES
    _right_press_age_frames: int = DRIFT_DOUBLE_TAP_WINDOW_FRAMES

    def reset(self) -> None:
        """Clear all step-to-step control history."""

        self._recent_boost_frames.clear()
        self._recent_boost_frame_sum = 0
        self._left_drift_held = False
        self._right_drift_held = False
        self._left_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES
        self._right_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES

    def record_step(self, *, control_state: ControllerState, frames_run: int) -> None:
        """Advance tracked control history by one env step."""

        frames_elapsed = max(int(frames_run), 0)
        boost_requested = bool(control_state.joypad_mask & BOOST_MASK)
        left_held = bool(control_state.joypad_mask & DRIFT_LEFT_MASK)
        right_held = bool(control_state.joypad_mask & DRIFT_RIGHT_MASK)

        self._record_recent_boost_pressure(
            boost_requested=boost_requested,
            frames_run=frames_elapsed,
        )
        self._left_press_age_frames = _advance_press_age(
            self._left_press_age_frames,
            was_held=self._left_drift_held,
            is_held=left_held,
            frames_elapsed=frames_elapsed,
        )
        self._right_press_age_frames = _advance_press_age(
            self._right_press_age_frames,
            was_held=self._right_drift_held,
            is_held=right_held,
            frames_elapsed=frames_elapsed,
        )
        self._left_drift_held = left_held
        self._right_drift_held = right_held

    def observation_fields(self) -> dict[str, float]:
        """Return control-history features passed into observation building."""

        return {
            "left_drift_held": float(self._left_drift_held),
            "right_drift_held": float(self._right_drift_held),
            # Normalize against the game's 15-frame double-tap window so the
            # policy can distinguish a fresh tap from an old one.
            "left_press_age_norm": _drift_press_age_norm(self._left_press_age_frames),
            "right_press_age_norm": _drift_press_age_norm(self._right_press_age_frames),
            "recent_boost_pressure": self._recent_boost_pressure(),
        }

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


def _advance_press_age(
    previous_age_frames: int,
    *,
    was_held: bool,
    is_held: bool,
    frames_elapsed: int,
) -> int:
    if is_held and not was_held:
        return min(frames_elapsed, DRIFT_DOUBLE_TAP_WINDOW_FRAMES)
    return min(previous_age_frames + frames_elapsed, DRIFT_DOUBLE_TAP_WINDOW_FRAMES)


def _drift_press_age_norm(frames: int) -> float:
    clamped_frames = min(max(int(frames), 0), DRIFT_DOUBLE_TAP_WINDOW_FRAMES)
    return clamped_frames / DRIFT_DOUBLE_TAP_WINDOW_FRAMES
