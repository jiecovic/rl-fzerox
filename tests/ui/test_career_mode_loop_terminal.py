# tests/ui/test_career_mode_loop_terminal.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from fzerox_emulator.arrays import Pcm16Samples, RgbFrame
from rl_fzerox.core.career_mode.controller import CareerControllerLifecycleEvents
from rl_fzerox.ui.watch.runtime.career_mode.loop.terminal import (
    observe_policy_race_terminal,
    observe_policy_start_terminal,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording import RecordingSegmentStatus


def test_policy_start_terminal_records_lifecycle_without_incrementing_episode() -> None:
    controller: Any = _TerminalController(
        observe_handled=True,
        policy_owns_control_after=False,
        has_active_attempt=True,
    )
    recorder: Any = _FrameRecorder()
    session: Any = object()
    info: dict[str, object] = {"game_mode": "gp_race_next_course"}

    terminal = observe_policy_start_terminal(
        controller=controller,
        session=session,
        info=info,
        active_policy_control=None,
        frame_recorder=recorder,
    )

    assert terminal is not None
    assert terminal.event == "policy_start_terminal"
    assert terminal.increment_episode is False
    assert terminal.viewer_info["terminal_observed"] is True
    assert terminal.lifecycle.recorded_event is True
    assert recorder.events == [terminal.viewer_info]


def test_policy_start_terminal_rejects_race_exit_without_terminal_result() -> None:
    controller: Any = _TerminalController(
        observe_handled=False,
        policy_owns_control_after=True,
        has_active_attempt=True,
    )
    session: Any = object()

    with pytest.raises(
        RuntimeError,
        match="Career Mode left a race before observing a game result",
    ):
        observe_policy_start_terminal(
            controller=controller,
            session=session,
            info={"game_mode": "gp_race_next_course"},
            active_policy_control=None,
            frame_recorder=None,
        )


def test_policy_race_terminal_waits_until_controller_releases_policy_control() -> None:
    still_racing: Any = _TerminalController(
        observe_handled=True,
        policy_owns_control_after=True,
        has_active_attempt=True,
    )
    session: Any = object()

    assert (
        observe_policy_race_terminal(
            controller=still_racing,
            session=session,
            info={"game_mode": "gp_race", "termination_reason": "finished"},
            active_policy_control=None,
            frame_recorder=None,
        )
        is None
    )

    terminal_controller: Any = _TerminalController(
        observe_handled=True,
        policy_owns_control_after=False,
        has_active_attempt=False,
    )
    recorder: Any = _FrameRecorder()

    terminal = observe_policy_race_terminal(
        controller=terminal_controller,
        session=session,
        info={"game_mode": "gp_race", "termination_reason": "finished"},
        active_policy_control=None,
        frame_recorder=recorder,
    )

    assert terminal is not None
    assert terminal.event == "race_terminal"
    assert terminal.increment_episode is True
    assert terminal.lifecycle.has_active_attempt is False
    assert recorder.events == [terminal.viewer_info]


class _TerminalController:
    def __init__(
        self,
        *,
        observe_handled: bool,
        policy_owns_control_after: bool,
        has_active_attempt: bool,
    ) -> None:
        self._observe_handled = observe_handled
        self._policy_owns_control_after = policy_owns_control_after
        self._has_active_attempt = has_active_attempt

    def observe_step(self, *, session: object, info: dict[str, object]) -> bool:
        _ = session
        if self._observe_handled:
            info["terminal_observed"] = True
        return self._observe_handled

    def policy_owns_control(self) -> bool:
        return self._policy_owns_control_after

    def viewer_info(
        self,
        *,
        info: dict[str, object],
        active_policy_control: object | None,
    ) -> dict[str, object]:
        viewed = dict(info)
        viewed["active_policy_control"] = active_policy_control
        return viewed

    def drain_lifecycle_events(self) -> CareerControllerLifecycleEvents:
        return CareerControllerLifecycleEvents(
            recording_close=None,
            emulator_reset_requested=False,
            has_active_attempt=self._has_active_attempt,
        )


class _FrameRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: Mapping[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None:
        _ = frame, info, audio_samples

    def record_event(self, *, info: Mapping[str, object]) -> None:
        self.events.append(dict(info))

    def finish_segment(
        self,
        *,
        status: RecordingSegmentStatus,
        info: Mapping[str, object],
    ) -> None:
        _ = status, info

    def drain_notices(self) -> tuple[str, ...]:
        return ()
