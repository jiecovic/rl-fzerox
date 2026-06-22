# src/rl_fzerox/core/career_mode/controller/lifecycle/recording.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.career_mode.navigation import MenuFacts
from rl_fzerox.core.manager.models import SaveAttemptStatus

CareerRecordingSegmentStatus = Literal["succeeded", "failed"]


def recording_status_from_attempt_status(
    status: SaveAttemptStatus | None,
) -> CareerRecordingSegmentStatus:
    if status == "succeeded":
        return "succeeded"
    return "failed"


@dataclass(frozen=True, slots=True)
class CareerRecordingSegmentClose:
    """FSM-owned signal that one cup-attempt recording segment should close."""

    status: CareerRecordingSegmentStatus
    info: dict[str, object]


@dataclass(slots=True)
class CareerRecordingSegmentTracker:
    """Track the controller-owned close signal for the current cup attempt.

    The progress FSM decides whether an attempt succeeded or failed. The
    recorder must not infer that from menu screens itself, because a replayed
    target can already be marked complete in the save file before the current
    attempt starts. This tracker only keeps small observed facts for diagnostics
    and emits the explicit close status supplied by the controller.
    """

    terminal_result_seen: bool = False
    failed_result_seen: bool = False
    post_gp_seen: bool = False
    pending_close: CareerRecordingSegmentClose | None = None

    def observe_terminal_result(self, info: dict[str, object]) -> None:
        reason = info.get("termination_reason")
        if reason not in {"finished", "retired", "crashed"}:
            return
        self.terminal_result_seen = True
        if reason in {"retired", "crashed"}:
            self.failed_result_seen = True

    def observe_progress_screen(self, facts: MenuFacts, info: dict[str, object]) -> None:
        del info
        if not self.terminal_result_seen:
            return
        if facts.is_post_gp_screen:
            self.post_gp_seen = True

    def close(
        self,
        *,
        status: CareerRecordingSegmentStatus,
        info: dict[str, object] | None = None,
    ) -> None:
        if self.pending_close is not None:
            return
        self.pending_close = CareerRecordingSegmentClose(status=status, info=dict(info or {}))
        self.reset()

    def force_close(
        self,
        *,
        status: CareerRecordingSegmentStatus,
        info: dict[str, object] | None = None,
    ) -> None:
        """Close the current segment when the FSM resets the emulator itself."""

        self.close(status=status, info=info)

    def pop_close(self) -> CareerRecordingSegmentClose | None:
        close = self.pending_close
        self.pending_close = None
        return close

    def reset(self) -> None:
        self.terminal_result_seen = False
        self.failed_result_seen = False
        self.post_gp_seen = False
