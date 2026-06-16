# src/rl_fzerox/core/career_mode/runner/controller/recording.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.career_mode.runner.menu import MenuFacts

CareerRecordingSegmentStatus = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class CareerRecordingSegmentClose:
    """FSM-owned signal that one cup-attempt recording segment should close."""

    status: CareerRecordingSegmentStatus


@dataclass(slots=True)
class CareerRecordingSegmentTracker:
    """Track visible game-flow facts for the current cup-attempt recording.

    This object deliberately knows nothing about manager DB progress. It only
    observes facts the live FSM sees on-screen: terminal course result, whether
    any terminal result failed, whether the post-GP success screen appeared,
    and whether the flow exited to menu/title/course-select/game-over.
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
        if not self.terminal_result_seen:
            return
        if facts.is_post_gp_screen:
            self.post_gp_seen = True
            if _post_gp_rank(info) not in {None, 1}:
                self.failed_result_seen = True
            return
        if not recording_segment_exit_screen(facts):
            return
        status: CareerRecordingSegmentStatus = (
            "failed" if self.failed_result_seen or not self.post_gp_seen else "succeeded"
        )
        self.pending_close = CareerRecordingSegmentClose(status=status)
        self.reset()

    def force_close(self, *, status: CareerRecordingSegmentStatus) -> None:
        """Close the current segment when the FSM resets the emulator itself."""

        self.pending_close = CareerRecordingSegmentClose(status=status)
        self.reset()

    def pop_close(self) -> CareerRecordingSegmentClose | None:
        close = self.pending_close
        self.pending_close = None
        return close

    def reset(self) -> None:
        self.terminal_result_seen = False
        self.failed_result_seen = False
        self.post_gp_seen = False


def recording_segment_exit_screen(facts: MenuFacts) -> bool:
    return (
        facts.is_title
        or facts.is_mode_select
        or facts.is_course_select
        or facts.game_mode == "game_over"
    )


def _post_gp_rank(info: dict[str, object]) -> int | None:
    value = info.get("position")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
