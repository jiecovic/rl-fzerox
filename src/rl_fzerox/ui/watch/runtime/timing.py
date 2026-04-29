# src/rl_fzerox/ui/watch/runtime/timing.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class _TimingLimits:
    min_fps: float = 1.0
    control_fps_adjust_step: float = 5.0


_TIMING_LIMITS = _TimingLimits()


class RateMeter:
    """Sliding-window event-rate meter for watch-loop diagnostics."""

    def __init__(self, *, window: int = 60) -> None:
        self._timestamps: deque[float] = deque(maxlen=max(2, int(window)))

    def tick(self, now: float | None = None) -> None:
        self._timestamps.append(time.perf_counter() if now is None else float(now))

    def rate_hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return float(len(self._timestamps) - 1) / elapsed

    def trim_to_recent(self, count: int = 2) -> None:
        keep = max(0, int(count))
        if keep <= 0 or len(self._timestamps) <= keep:
            return
        recent = list(self._timestamps)[-keep:]
        self._timestamps.clear()
        self._timestamps.extend(recent)


def _resolve_control_fps(
    setting: object,
    *,
    native_control_fps: float,
) -> float | None:
    """Resolve configured env-step FPS; `None` means uncapped fast-forward."""

    if setting in (None, "auto"):
        return max(native_control_fps, _TIMING_LIMITS.min_fps)
    if setting == "unlimited":
        return None
    if isinstance(setting, int | float):
        return max(float(setting), _TIMING_LIMITS.min_fps)
    raise ValueError(f"Unsupported watch control_fps value: {setting!r}")


def _resolve_render_fps(setting: object, *, native_fps: float) -> float | None:
    """Resolve pygame redraw FPS; `None` means uncapped redraw."""

    if setting is None:
        return 60.0
    if setting == "auto":
        return max(native_fps, _TIMING_LIMITS.min_fps)
    if setting == "unlimited":
        return None
    if isinstance(setting, int | float):
        return max(float(setting), _TIMING_LIMITS.min_fps)
    raise ValueError(f"Unsupported watch render_fps value: {setting!r}")


def _target_seconds(target_fps: float | None) -> float | None:
    if target_fps is None:
        return None
    return 1.0 / max(target_fps, _TIMING_LIMITS.min_fps)


def _adjust_control_fps(
    target_control_fps: float | None,
    delta: int,
    *,
    native_control_fps: float | None,
) -> float | None:
    if target_control_fps is None and delta > 0:
        return None
    base_fps = (
        native_control_fps
        if target_control_fps is None and native_control_fps is not None
        else target_control_fps
    )
    if base_fps is None:
        base_fps = _TIMING_LIMITS.min_fps
    return max(
        _TIMING_LIMITS.min_fps,
        base_fps + (delta * _TIMING_LIMITS.control_fps_adjust_step),
    )
