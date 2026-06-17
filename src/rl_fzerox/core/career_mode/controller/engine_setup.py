# src/rl_fzerox/core/career_mode/controller/engine_setup.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rl_fzerox.core.career_mode.controller.menu_flow import engine_adjust_tap_count
from rl_fzerox.core.career_mode.navigation import (
    MENU_TIMING,
    MenuFacts,
    MenuInput,
    RawMenuStep,
    course_id_from_info,
    engine_adjust_steps,
    raw_step,
)
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup


@dataclass(slots=True)
class CareerEngineSetupFlow:
    """Own menu-side engine-slider adjustment before a GP race starts."""

    applied_course_id: str | None = None
    adjust_taps: int = 0
    ready_course_id: str | None = None
    ready_target: int | None = None
    ready_frames: int = 0

    def applied_for(self, course_id: str) -> bool:
        return self.applied_course_id == course_id

    def reset_adjustment(self) -> None:
        self.applied_course_id = None
        self.adjust_taps = 0
        self.reset_confirmation()

    def reset_confirmation(self) -> None:
        self.ready_course_id = None
        self.ready_target = None
        self.ready_frames = 0

    def reset_tap_budget(self) -> None:
        self.adjust_taps = 0
        self.reset_confirmation()

    def next_step(
        self,
        *,
        info: dict[str, object],
        course_setup: ManagedSaveCourseSetup | None,
        queue_menu_steps: Callable[[tuple[RawMenuStep, ...]], RawMenuStep],
    ) -> RawMenuStep:
        facts = MenuFacts.from_info(info)
        if not facts.is_machine_settings:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_settings")
        course_id = course_id_from_info(info)
        if course_id is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_course")
        if course_setup is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_setup")

        current = facts.engine_setting_raw_value
        target = course_setup.engine_setting_raw_value
        if current == target:
            self.adjust_taps = 0
            if self.ready_course_id == course_id and self.ready_target == target:
                self.ready_frames += 1
            else:
                self.ready_course_id = course_id
                self.ready_target = target
                self.ready_frames = 1
            if self.ready_frames >= MENU_TIMING.engine_ready_confirm_frames:
                self.applied_course_id = course_id
                return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:ready")
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:confirm")

        self.reset_confirmation()
        if current is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_read")
        remaining_taps = MENU_TIMING.max_engine_adjust_taps - self.adjust_taps
        if remaining_taps <= 0:
            raise RuntimeError(
                f"Career Mode could not reach the requested engine setting {target} from {current}"
            )
        steps = engine_adjust_steps(
            current=current,
            target=target,
            max_taps=remaining_taps,
        )
        self.adjust_taps += engine_adjust_tap_count(steps)
        return queue_menu_steps(steps)
