# src/rl_fzerox/core/envs/engine/controls/pressure.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rl_fzerox.core.envs.observations import DEFAULT_STATE_VECTOR_SPEC


@dataclass(slots=True)
class RecentControlPressure:
    """Track short-window button/axis pressure features for observations."""

    _boost_frames: deque[int] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames),
    )
    _steer_frames: deque[float] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames),
    )
    _boost_frame_sum: int = 0
    _steer_frame_sum: float = 0.0

    def reset(self) -> None:
        self._boost_frames.clear()
        self._steer_frames.clear()
        self._boost_frame_sum = 0
        self._steer_frame_sum = 0.0

    def record_boost(self, *, requested: bool, frames_run: int) -> None:
        encoded = 1 if requested else 0
        for _ in range(frames_run):
            if len(self._boost_frames) == self._boost_frames.maxlen:
                removed = self._boost_frames.popleft()
                self._boost_frame_sum -= removed
            self._boost_frames.append(encoded)
            self._boost_frame_sum += encoded

    def boost_pressure(self) -> float:
        if not self._boost_frames:
            return 0.0
        return self._boost_frame_sum / len(self._boost_frames)

    def record_steer(self, *, axis: float, frames_run: int) -> None:
        for _ in range(frames_run):
            if len(self._steer_frames) == self._steer_frames.maxlen:
                removed = self._steer_frames.popleft()
                self._steer_frame_sum -= removed
            self._steer_frames.append(axis)
            self._steer_frame_sum += axis

    def steer_pressure(self) -> float:
        if not self._steer_frames:
            return 0.0
        window_size = self._steer_frames.maxlen
        if window_size is None or window_size <= 0:
            return 0.0
        return self._steer_frame_sum / window_size
