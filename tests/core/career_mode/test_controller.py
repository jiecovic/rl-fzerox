# tests/core/career_mode/test_controller.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_fzerox.core.career_mode.runner.controller import (
    CareerModeController,
    _cup_selection_input,
)
from rl_fzerox.core.career_mode.runner.menu import (
    MENU_TIMING,
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    ObservedMenuScreen,
    engine_adjust_steps,
    observed_menu_screen,
)
from rl_fzerox.core.career_mode.runner.terminal import post_terminal_progress_screen
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def test_cup_selection_moves_left_when_target_is_before_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=3, target_cup_index=0) is MenuInput.LEFT


def test_cup_selection_moves_right_when_target_is_after_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=0, target_cup_index=3) is MenuInput.RIGHT


def test_cup_selection_moves_right_until_selected_cup_is_known() -> None:
    assert _cup_selection_input(selected_cup_index=None, target_cup_index=0) is MenuInput.RIGHT


def test_engine_adjust_steps_use_fast_bounded_right_burst() -> None:
    steps = engine_adjust_steps(current=0, target=100)

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


def test_policy_race_continues_terminal_gp_result_screen(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._phase = CareerPhase.POLICY_RACE

    step = controller.next_raw_step(
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert step is not None
    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase == "continue_after_race:accept:1"
    assert controller.phase is CareerPhase.CONTINUE_AFTER_RACE

    settle_step = controller.next_raw_step(
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert settle_step is not None
    assert settle_step.menu_input is MenuInput.NEUTRAL
    assert settle_step.phase == "continue_after_race:accept:1:settle"


def test_unskippable_credits_are_post_gp_screen() -> None:
    screen = observed_menu_screen(
        MenuFacts.from_info({"game_mode": "unskippable_credits"}),
        difficulty_popup_state=DifficultyPopupState.CLOSED,
    )

    assert screen is ObservedMenuScreen.POST_GP


def test_returned_main_menu_is_post_terminal_progress_screen() -> None:
    assert post_terminal_progress_screen(MenuFacts.from_info({"game_mode": "main_menu"}))


def test_no_active_attempt_rejects_menu_navigation(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._progress._attempt_id = None
    controller._phase = CareerPhase.WAIT_FOR_GP_RACE

    with pytest.raises(RuntimeError, match="no active save attempt"):
        controller.next_raw_step(
            info={"game_mode": "gp_race", "termination_reason": "finished"},
        )


def test_checkpoint_refresh_is_armed_only_after_finished_attempt(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    resolver = _PolicyResolverStub()
    controller.__dict__["_policy_resolver"] = resolver

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False]

    controller._remember_finished_attempt(
        SimpleNamespace(
            finished_attempt_id="attempt-a",
            finished_status="failed",
            finished_failure_reason=None,
        )
    )

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False, True]

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False, True, False]


class _PolicyResolverStub:
    def __init__(self) -> None:
        self.refresh_requests: list[bool] = []

    def resolve(
        self,
        info: dict[str, object],
        *,
        refresh_artifact: bool = False,
    ) -> object:
        del info
        self.refresh_requests.append(refresh_artifact)
        return SimpleNamespace(
            activated_new_policy=False,
            camera_setting=None,
            control="control",
        )


def _controller(tmp_path: Path) -> CareerModeController:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    save_game = store.create_save_game(
        name="Career Save",
        save_games_root=tmp_path / "save-games",
    )
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    return CareerModeController(
        _race_setup(),
        db_path=db_path,
        save_game_id=save_game.id,
        attempt_id=attempt.id,
        device="cpu",
    )


def _race_setup() -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="novice",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_raw_value=50,
    )
