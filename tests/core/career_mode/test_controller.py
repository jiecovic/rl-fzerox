# tests/core/career_mode/test_controller.py
from __future__ import annotations

from rl_fzerox.core.career_mode.runner.controller import _cup_selection_input
from rl_fzerox.core.career_mode.runner.menu import MenuInput


def test_cup_selection_moves_left_when_target_is_before_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=3, target_cup_index=0) is MenuInput.LEFT


def test_cup_selection_moves_right_when_target_is_after_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=0, target_cup_index=3) is MenuInput.RIGHT


def test_cup_selection_moves_right_until_selected_cup_is_known() -> None:
    assert _cup_selection_input(selected_cup_index=None, target_cup_index=0) is MenuInput.RIGHT
