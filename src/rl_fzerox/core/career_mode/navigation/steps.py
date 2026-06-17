# src/rl_fzerox/core/career_mode/navigation/steps.py
from __future__ import annotations

from rl_fzerox.core.career_mode.navigation.types import (
    MENU_TIMING,
    CareerPhase,
    MenuInput,
    RawMenuStep,
)
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def machine_select_steps(
    setup: CareerModeRaceSetupConfig,
) -> tuple[RawMenuStep, ...]:
    steps: list[RawMenuStep] = []
    for tap_index in range(setup.machine_select_row):
        steps.extend(
            tap_steps(
                MenuInput.DOWN,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase=f"select_machine:down:{tap_index + 1}",
            )
        )
    for tap_index in range(setup.machine_select_column):
        steps.extend(
            tap_steps(
                MenuInput.RIGHT,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase=f"select_machine:right:{tap_index + 1}",
            )
        )
    return tuple(steps)


def engine_adjust_steps(
    *,
    current: int,
    target: int,
    max_taps: int | None = None,
) -> tuple[RawMenuStep, ...]:
    """Return a bounded burst of engine slider taps toward the target value."""

    delta = target - current
    if delta == 0:
        return ()
    if max_taps is not None and max_taps <= 0:
        return ()
    menu_input = MenuInput.RIGHT if delta > 0 else MenuInput.LEFT
    tap_count = min(
        abs(delta),
        MENU_TIMING.engine_adjust_max_taps_per_burst,
        max_taps if max_taps is not None else abs(delta),
    )
    steps: list[RawMenuStep] = []
    for tap_index in range(tap_count):
        steps.extend(
            tap_steps(
                menu_input,
                hold_frames=MENU_TIMING.engine_adjust_hold_frames,
                settle_frames=MENU_TIMING.engine_adjust_settle_frames,
                phase=f"apply_engine:{menu_input.value}:{tap_index + 1}/{tap_count}",
            )
        )
    return tuple(steps)


def continue_after_race_step(press_index: int) -> RawMenuStep:
    """Return one post-race continuation pulse after a terminal edge."""

    return raw_step(
        MenuInput.ACCEPT,
        MENU_TIMING.result_continue_hold_frames,
        phase=f"continue_after_race:accept:{press_index + 1}",
    )


def continue_next_course_step() -> RawMenuStep:
    """Advance the GP next-course screen to machine settings."""

    return raw_step(
        MenuInput.ACCEPT,
        MENU_TIMING.menu_hold_frames,
        phase="continue_after_race:next_course_accept",
    )


def tap_steps(
    menu_input: MenuInput,
    *,
    hold_frames: int,
    settle_frames: int,
    phase: str,
) -> tuple[RawMenuStep, ...]:
    return (
        raw_step(menu_input, hold_frames, phase=phase),
        raw_step(MenuInput.NEUTRAL, settle_frames, phase=f"{phase}:settle"),
    )


def raw_step(
    menu_input: MenuInput,
    frames: int,
    *,
    phase: str,
) -> RawMenuStep:
    return RawMenuStep(menu_input=menu_input, frames=frames, phase=phase)


def phase_from_step(step: RawMenuStep) -> CareerPhase:
    if step.phase.startswith("select_difficulty"):
        return CareerPhase.SELECT_DIFFICULTY
    if step.phase.startswith("enter_course_select"):
        return CareerPhase.ENTER_COURSE_SELECT
    if step.phase.startswith("select_cup"):
        return CareerPhase.SELECT_CUP
    if step.phase.startswith("enter_machine_select"):
        return CareerPhase.ENTER_MACHINE_SELECT
    if step.phase.startswith("continue_after_race"):
        return CareerPhase.CONTINUE_AFTER_RACE
    if step.phase.startswith("select_machine"):
        return CareerPhase.SELECT_MACHINE
    if step.phase.startswith("enter_machine_settings"):
        return CareerPhase.ENTER_MACHINE_SETTINGS
    if step.phase.startswith("apply_engine"):
        return CareerPhase.APPLY_ENGINE
    if step.phase.startswith("enter_race"):
        return CareerPhase.ENTER_RACE
    return CareerPhase.BOOT_TO_DIFFICULTY
