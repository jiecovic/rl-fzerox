# src/rl_fzerox/core/career_mode/controller/menu_flow.py
from __future__ import annotations

from rl_fzerox.core.career_mode.navigation import (
    MenuInput,
    ObservedMenuScreen,
    RawMenuStep,
)


def is_neutral_settle_step(step: RawMenuStep) -> bool:
    return step.menu_input is MenuInput.NEUTRAL and step.phase.endswith(":settle")


def pending_step_matches_observed_screen(
    step: RawMenuStep,
    screen: ObservedMenuScreen,
) -> bool:
    if step.menu_input is MenuInput.NEUTRAL:
        return True
    if step.phase.startswith("title_to_main_menu"):
        return screen is ObservedMenuScreen.TITLE
    if step.phase.startswith("main_menu"):
        return screen in {
            ObservedMenuScreen.MAIN_MENU_GP,
            ObservedMenuScreen.MAIN_MENU_OTHER,
        }
    if step.phase.startswith("select_difficulty"):
        return screen is ObservedMenuScreen.DIFFICULTY_POPUP
    if step.phase.startswith("enter_course_select:confirm_difficulty"):
        return screen is ObservedMenuScreen.DIFFICULTY_CONFIRM
    if step.phase.startswith(("select_cup", "enter_machine_select")):
        return screen is ObservedMenuScreen.COURSE_SELECT
    if step.phase.startswith("select_machine"):
        return screen is ObservedMenuScreen.MACHINE_SELECT
    if step.phase.startswith("enter_machine_settings"):
        return screen is ObservedMenuScreen.MACHINE_SELECT
    if step.phase.startswith(("apply_engine", "enter_race")):
        return screen is ObservedMenuScreen.MACHINE_SETTINGS
    if step.phase.startswith("course_select:wrong_difficulty"):
        return screen is ObservedMenuScreen.COURSE_SELECT
    if step.phase.startswith("continue_after_race"):
        return screen in {
            ObservedMenuScreen.GP_RACE,
            ObservedMenuScreen.RESULTS,
            ObservedMenuScreen.GP_NEXT_COURSE,
            ObservedMenuScreen.POST_GP,
        }
    return False


def cup_selection_input(*, selected_cup_index: int | None, target_cup_index: int) -> MenuInput:
    if selected_cup_index is None or selected_cup_index < target_cup_index:
        return MenuInput.RIGHT
    return MenuInput.LEFT


def engine_adjust_tap_count(steps: tuple[RawMenuStep, ...]) -> int:
    return sum(step.menu_input is not MenuInput.NEUTRAL for step in steps)
