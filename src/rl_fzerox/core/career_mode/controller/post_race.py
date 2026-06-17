# src/rl_fzerox/core/career_mode/controller/post_race.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation import (
    MENU_TIMING,
    MenuFacts,
    MenuInput,
    RawMenuStep,
    continue_after_race_step,
    raw_step,
)


@dataclass(slots=True)
class PostRaceContinuation:
    """Track visible flow after a terminal race result.

    This state does not decide success or failure. It only owns the small
    continuation counters needed to advance result screens and detect when a
    fresh next race is ready for policy handoff.
    """

    awaiting_new_race_after_terminal: bool = False
    continuing_race_result: bool = False
    observed_terminal_race_result: bool = False
    continue_pulses: int = 0
    last_progress_sync_continue_pulse: int = -1
    fresh_race_ready_frames: int = 0

    def enter_continue_after_race(self) -> None:
        self.awaiting_new_race_after_terminal = True
        self.continuing_race_result = True
        self.observed_terminal_race_result = True
        self.continue_pulses = 0
        self.last_progress_sync_continue_pulse = -1
        self.fresh_race_ready_frames = 0

    def reset(self) -> None:
        self.awaiting_new_race_after_terminal = False
        self.continuing_race_result = False
        self.observed_terminal_race_result = False
        self.continue_pulses = 0
        self.last_progress_sync_continue_pulse = -1
        self.fresh_race_ready_frames = 0

    def observe_terminal_result(self) -> None:
        self.observed_terminal_race_result = True
        self.continuing_race_result = True

    def continue_observed_result(self) -> bool:
        if not self.observed_terminal_race_result:
            return False
        self.continuing_race_result = True
        return True

    def stop_result_continuation(self) -> None:
        self.continuing_race_result = False

    def mark_progress_synced(self) -> None:
        self.last_progress_sync_continue_pulse = self.continue_pulses

    def new_race_ready_for_policy(self, info: dict[str, object]) -> bool:
        facts = MenuFacts.from_info(info)
        if not self.awaiting_new_race_after_terminal:
            return not facts.terminal_race_result

        fresh_race_ready = facts.fresh_race_ready_for_policy or (
            self.observed_terminal_race_result and facts.has_fresh_race_shape
        )
        if not fresh_race_ready:
            self.fresh_race_ready_frames = 0
            return False

        self.fresh_race_ready_frames += 1
        if self.fresh_race_ready_frames < MENU_TIMING.fresh_race_ready_frames:
            return False

        self.awaiting_new_race_after_terminal = False
        self.continuing_race_result = False
        self.observed_terminal_race_result = False
        self.fresh_race_ready_frames = 0
        return True

    def continue_after_race_pulse(self) -> tuple[RawMenuStep, RawMenuStep]:
        step = continue_after_race_step(self.continue_pulses)
        self.continue_pulses += 1
        settle_step = raw_step(
            MenuInput.NEUTRAL,
            MENU_TIMING.result_continue_settle_frames,
            phase=f"{step.phase}:settle",
        )
        return step, settle_step
