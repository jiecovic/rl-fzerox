# tests/core/career_mode/test_controller.py
from __future__ import annotations

from rl_fzerox.core.career_mode.runner.controller import _cup_selection_input
from rl_fzerox.core.career_mode.runner.menu import MENU_TIMING, MenuInput, engine_adjust_steps


def test_cup_selection_moves_left_when_target_is_before_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=3, target_cup_index=0) is MenuInput.LEFT


def test_cup_selection_moves_right_when_target_is_after_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=0, target_cup_index=3) is MenuInput.RIGHT


def test_cup_selection_moves_right_until_selected_cup_is_known() -> None:
    assert _cup_selection_input(selected_cup_index=None, target_cup_index=0) is MenuInput.RIGHT


def test_engine_adjust_steps_use_fast_bounded_right_burst() -> None:
    steps = engine_adjust_steps(current=40, target=60)

    active_steps = [step for step in steps if step.menu_input is not MenuInput.NEUTRAL]
    settle_steps = [step for step in steps if step.menu_input is MenuInput.NEUTRAL]

    assert len(active_steps) == MENU_TIMING.engine_adjust_max_taps_per_burst
    assert len(settle_steps) == MENU_TIMING.engine_adjust_max_taps_per_burst
    assert {step.menu_input for step in active_steps} == {MenuInput.RIGHT}
    assert {step.frames for step in active_steps} == {MENU_TIMING.engine_adjust_hold_frames}
    assert {step.frames for step in settle_steps} == {MENU_TIMING.engine_adjust_settle_frames}


def test_engine_adjust_steps_cap_burst_by_remaining_safety_budget() -> None:
    steps = engine_adjust_steps(current=100, target=0, max_taps=3)

    active_steps = [step for step in steps if step.menu_input is not MenuInput.NEUTRAL]

    assert len(active_steps) == 3
    assert {step.menu_input for step in active_steps} == {MenuInput.LEFT}


def test_engine_adjust_steps_empty_when_target_already_selected() -> None:
    assert engine_adjust_steps(current=50, target=50) == ()
