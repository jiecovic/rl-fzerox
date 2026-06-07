# src/rl_fzerox/ui/watch/runtime/career_mode/timing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.timing import _target_seconds


@dataclass(frozen=True, slots=True)
class PolicyTiming:
    """Control-rate target for a policy-owned Career Mode race."""

    target_fps: float | None
    target_seconds: float | None


class ControlTimingSession(Protocol):
    target_control_fps: float | None
    target_control_seconds: float | None


class PolicyTimingSession(Protocol):
    native_fps: float

    def snapshot_config(self, base_config: WatchAppConfig) -> WatchAppConfig: ...


def native_frame_seconds(target_control_seconds: float | None) -> float | None:
    return target_control_seconds


def active_policy_timing(
    config: WatchAppConfig,
    session: PolicyTimingSession,
    *,
    native_control_fps: float,
    target_control_fps: float | None,
) -> PolicyTiming:
    snapshot_config = session.snapshot_config(config)
    policy_native_control_fps = session.native_fps / max(
        1,
        snapshot_config.env.action_repeat,
    )
    if target_control_fps is None:
        target_fps = None
    else:
        speed_multiplier = target_control_fps / max(native_control_fps, 1e-9)
        target_fps = max(1.0, policy_native_control_fps * speed_multiplier)
    return PolicyTiming(
        target_fps=target_fps,
        target_seconds=_target_seconds(target_fps),
    )


def snapshot_target_control_fps(
    *,
    config: WatchAppConfig,
    session: PolicyTimingSession,
    native_control_fps: float,
    target_control_fps: float | None,
    policy_active: bool,
) -> float | None:
    if not policy_active:
        return target_control_fps
    return active_policy_timing(
        config,
        session,
        native_control_fps=native_control_fps,
        target_control_fps=target_control_fps,
    ).target_fps


def snapshot_action_repeat(
    config: WatchAppConfig,
    *,
    policy_active: bool,
) -> int:
    if not policy_active:
        return 1
    return max(1, int(config.env.action_repeat))


def set_session_control_timing(
    session: ControlTimingSession,
    *,
    target_control_fps: float | None,
    target_control_seconds: float | None,
) -> None:
    session.target_control_fps = target_control_fps
    session.target_control_seconds = target_control_seconds


def measured_game_fps(*, control_fps: float, action_repeat: int) -> float:
    if control_fps <= 0.0:
        return 0.0
    return control_fps * max(1, int(action_repeat))


def target_game_fps(
    *,
    target_control_fps: float | None,
    action_repeat: int,
) -> float | None:
    if target_control_fps is None:
        return None
    return max(0.0, float(target_control_fps) * max(1, int(action_repeat)))


def with_measured_game_fps(
    info: dict[str, object],
    *,
    game_fps: float,
    game_fps_target: float | None,
) -> dict[str, object]:
    if game_fps <= 0.0:
        return info if game_fps_target is None else info | {"game_fps_target": game_fps_target}
    next_info = info | {"game_fps": game_fps}
    if game_fps_target is not None:
        next_info["game_fps_target"] = game_fps_target
    return next_info
