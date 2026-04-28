# src/rl_fzerox/core/envs/engine/controls/boost.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BoostTimingState:
    """Track manual-boost decision cadence and request lockout."""

    decision_interval_frames: int = 1
    request_lockout_frames: int = 0
    episode_frame_index: int = 0
    _next_decision_frame: int = 0
    _request_lockout_remaining_frames: int = 0

    def reset(self) -> None:
        self.episode_frame_index = 0
        self._next_decision_frame = 0
        self._request_lockout_remaining_frames = 0

    def action_allowed(self) -> bool:
        """Return whether this frame is a tactical manual-boost decision slot."""

        if self._request_lockout_remaining_frames > 0:
            return False
        interval = max(int(self.decision_interval_frames), 1)
        return interval <= 1 or self.episode_frame_index >= self._next_decision_frame

    def record(self, *, boost_requested: bool, frames_elapsed: int) -> None:
        """Advance boost timing state by one env step."""

        lockout_frames = max(int(self.request_lockout_frames), 0)
        boost_decision_slot_was_open = self.episode_frame_index >= self._next_decision_frame
        if boost_requested and lockout_frames > 0:
            self._request_lockout_remaining_frames = max(lockout_frames - frames_elapsed, 0)
        else:
            self._request_lockout_remaining_frames = max(
                self._request_lockout_remaining_frames - frames_elapsed,
                0,
            )
        if boost_decision_slot_was_open:
            interval = max(int(self.decision_interval_frames), 1)
            while self._next_decision_frame <= self.episode_frame_index:
                self._next_decision_frame += interval
        self.episode_frame_index += frames_elapsed
