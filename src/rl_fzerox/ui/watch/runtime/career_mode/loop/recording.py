# src/rl_fzerox/ui/watch/runtime/career_mode/loop/recording.py
from __future__ import annotations

from rl_fzerox.core.career_mode.runner.controller import CareerModeController
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder


def drain_recording_notices(frame_recorder: FrameRecorder | None) -> tuple[str, ...]:
    if frame_recorder is None:
        return ()
    return frame_recorder.drain_notices()


def record_controller_event(
    *,
    controller: CareerModeController,
    frame_recorder: FrameRecorder | None,
    info: dict[str, object],
) -> None:
    """Record one FSM event, closing the segment when the FSM says it ended.

    The controller owns cup-attempt lifecycle because it sees the menu/race FSM.
    The recorder owns only bytes on disk. This bridge is the single place where
    an FSM close signal becomes a recorder `finish_segment()` side effect.
    """

    if frame_recorder is None:
        return
    close = controller.pop_recording_segment_close()
    if close is None:
        frame_recorder.record_event(info=info)
        return
    frame_recorder.finish_segment(status=close.status, info=info)


def finish_pending_recording_segment(
    *,
    controller: CareerModeController,
    frame_recorder: FrameRecorder | None,
    info: dict[str, object],
) -> None:
    """Consume a deferred segment close after menu-progress FSM polling.

    `CareerModeController.before_step()` can observe the exit screen before the
    worker records a normal terminal event. This keeps that path explicit
    without making the recorder infer lifecycle from DB progress fields.
    """

    if frame_recorder is None:
        return
    close = controller.pop_recording_segment_close()
    if close is not None:
        frame_recorder.finish_segment(status=close.status, info=info)
