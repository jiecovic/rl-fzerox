# src/rl_fzerox/ui/watch/runtime/career_mode/loop/recording.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder


def drain_recording_notices(frame_recorder: FrameRecorder | None) -> tuple[str, ...]:
    if frame_recorder is None:
        return ()
    return frame_recorder.drain_notices()


@dataclass(frozen=True, slots=True)
class ControllerLifecycleResult:
    reset_requested: bool
    has_active_attempt: bool
    recording_close_status: str | None = None
    recorded_event: bool = False


def handle_controller_lifecycle(
    *,
    controller: CareerModeController,
    frame_recorder: FrameRecorder | None,
    info: dict[str, object],
    record_event: bool = False,
) -> ControllerLifecycleResult:
    """Drain controller lifecycle signals and apply runtime side effects.

    The controller owns cup-attempt lifecycle because it sees the menu/race FSM.
    The recorder and emulator live in the watch runtime. Draining all lifecycle
    signals in one place prevents recording, reset, and runner-exit decisions
    from being consumed by different branches.
    """

    events = controller.drain_lifecycle_events()
    recording_close_status = None
    recorded_event = False
    if frame_recorder is not None and events.recording_close is not None:
        frame_recorder.finish_segment(status=events.recording_close.status, info=info)
        recording_close_status = events.recording_close.status
    elif frame_recorder is not None and record_event:
        frame_recorder.record_event(info=info)
        recorded_event = True
    return ControllerLifecycleResult(
        reset_requested=events.emulator_reset_requested,
        has_active_attempt=events.has_active_attempt,
        recording_close_status=recording_close_status,
        recorded_event=recorded_event,
    )
