# src/rl_fzerox/core/career_mode/runner/controller/recording.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.career_mode.runner.menu import POST_GP_RECORDING_END_MODES, MenuFacts

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
    any terminal result failed, whether the post-GP success ceremony appeared,
    and whether the flow exited to menu/title/course-select/game-over.

    `gp_end_cutscene` is the winning ceremony, so it is part of the attempt
    recording. Credits or a return to menu/title/course-select is the segment
    boundary. Some credits screens never return to the menu, so the controller
    may force an emulator reset after the boundary is observed.
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
            if self.pending_close is not None:
                return
            self.post_gp_seen = True
            final_rank = _post_gp_final_rank(info)
            if final_rank not in {None, 1}:
                self.failed_result_seen = True
                self.pending_close = CareerRecordingSegmentClose(status="failed")
                self.reset()
                return
            if facts.game_mode not in POST_GP_RECORDING_END_MODES:
                return
            status: CareerRecordingSegmentStatus = (
                "failed" if self.failed_result_seen else "succeeded"
            )
            self.pending_close = CareerRecordingSegmentClose(status=status)
            self.reset()
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

        if self.pending_close is not None:
            return
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


def _post_gp_final_rank(info: dict[str, object]) -> int | None:
    value = info.get("career_mode_gp_final_rank", info.get("gp_final_rank"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
