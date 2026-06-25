# src/rl_fzerox/ui/watch/runtime/career_mode/loop/terminal.py
"""Terminal race observation helpers for the Career Mode watch loop.

The Career Mode controller owns the decision that a policy race reached a
terminal result. The watch loop still owns runtime side effects such as
snapshot refresh, emulator reset, and idle stepping. This module keeps the
controller observation and lifecycle-drain sequence in one place without moving
the runner's mutable runtime state across module boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.career_mode.navigation import in_gp_race
from rl_fzerox.core.career_mode.policy import CareerModePolicyControl
from rl_fzerox.ui.watch.runtime.career_mode.loop.recording import (
    ControllerLifecycleResult,
    handle_controller_lifecycle,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession

PolicyTerminalEvent = Literal["policy_start_terminal", "race_terminal"]


@dataclass(frozen=True, slots=True)
class ObservedPolicyTerminal:
    """Controller terminal result plus the paired lifecycle side effects."""

    event: PolicyTerminalEvent
    viewer_info: dict[str, object]
    lifecycle: ControllerLifecycleResult
    increment_episode: bool


def observe_policy_start_terminal(
    *,
    controller: CareerModeController,
    session: CareerModeRuntimeSession,
    info: dict[str, object],
    active_policy_control: CareerModePolicyControl | None,
    frame_recorder: FrameRecorder | None,
) -> ObservedPolicyTerminal | None:
    """Observe a terminal result before policy control has started.

    This path handles edge cases where the game leaves the race during the
    intro/countdown handoff. If the race was exited without a controller-visible
    terminal result, keeping the old hard failure is intentional: continuing
    would desync controller progress from the emulator screen.
    """

    terminal_handled = controller.observe_step(session=session, info=info)
    if not terminal_handled:
        if not in_gp_race(info):
            raise RuntimeError("Career Mode left a race before observing a game result")
        return None
    return _observed_policy_terminal(
        controller=controller,
        frame_recorder=frame_recorder,
        info=info,
        active_policy_control=active_policy_control,
        event="policy_start_terminal",
        increment_episode=False,
    )


def observe_policy_race_terminal(
    *,
    controller: CareerModeController,
    session: CareerModeRuntimeSession,
    info: dict[str, object],
    active_policy_control: CareerModePolicyControl | None,
    frame_recorder: FrameRecorder | None,
) -> ObservedPolicyTerminal | None:
    """Observe a terminal result after a policy/manual race step."""

    terminal_handled = controller.observe_step(session=session, info=info)
    if not terminal_handled or controller.policy_owns_control():
        return None
    return _observed_policy_terminal(
        controller=controller,
        frame_recorder=frame_recorder,
        info=info,
        active_policy_control=active_policy_control,
        event="race_terminal",
        increment_episode=True,
    )


def _observed_policy_terminal(
    *,
    controller: CareerModeController,
    frame_recorder: FrameRecorder | None,
    info: dict[str, object],
    active_policy_control: CareerModePolicyControl | None,
    event: PolicyTerminalEvent,
    increment_episode: bool,
) -> ObservedPolicyTerminal:
    viewer_info = controller.viewer_info(
        info=info,
        active_policy_control=active_policy_control,
    )
    lifecycle = handle_controller_lifecycle(
        controller=controller,
        frame_recorder=frame_recorder,
        info=viewer_info,
        record_event=True,
    )
    return ObservedPolicyTerminal(
        event=event,
        viewer_info=viewer_info,
        lifecycle=lifecycle,
        increment_episode=increment_episode,
    )
