# src/rl_fzerox/core/envs/observations/stack.py
from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces


class FrameStackBuffer:
    """Keep a channels-last stack of recent observation frames."""

    def __init__(self, *, frame_space: spaces.Box, frame_stack: int) -> None:
        if frame_space.dtype != np.uint8:
            raise ValueError("FrameStackBuffer currently expects uint8 image observations")

        self._frame_stack = frame_stack
        self._frame_shape = tuple(int(value) for value in frame_space.shape)
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self._frame_shape[0],
                self._frame_shape[1],
                self._frame_shape[2] * frame_stack,
            ),
            dtype=np.uint8,
        )

    @property
    def observation_space(self) -> spaces.Box:
        """Return the stacked observation space."""

        return self._observation_space

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Return the shape of one processed observation frame."""

        height, width, channels = self._frame_shape
        return height, width, channels

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Fill the stack with one repeated processed frame."""

        normalized = self._normalize_frame(frame)
        self._frames.clear()
        for _ in range(self._frame_stack):
            self._frames.append(normalized.copy())
        return self.observation()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """Append one processed frame and return the stacked observation."""

        normalized = self._normalize_frame(frame)
        if not self._frames:
            return self.reset(normalized)
        self._frames.append(normalized)
        return self.observation()

    def observation(self) -> np.ndarray:
        """Return the current channels-last stacked observation."""

        if len(self._frames) != self._frame_stack:
            raise RuntimeError("Frame stack is not initialized")
        return np.ascontiguousarray(np.concatenate(tuple(self._frames), axis=2))

    def latest_frame(self) -> np.ndarray:
        """Return the newest processed frame in the stack."""

        if not self._frames:
            raise RuntimeError("Frame stack is not initialized")
        return np.ascontiguousarray(self._frames[-1])

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            raise ValueError("FrameStackBuffer expects uint8 frames")
        if tuple(int(value) for value in frame.shape) != self._frame_shape:
            raise ValueError(
                f"Expected frame shape {self._frame_shape}, got {tuple(frame.shape)!r}"
            )
        return np.ascontiguousarray(frame)
