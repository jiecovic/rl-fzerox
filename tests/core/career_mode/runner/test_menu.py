# tests/core/career_mode/runner/test_menu.py

from __future__ import annotations

import pytest

from rl_fzerox.core.career_mode.runner.menu import (
    MENU_TIMING,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    ObservedMenuScreen,
    continue_after_race_step,
    continue_next_course_step,
    machine_select_steps,
    observed_menu_screen,
    select_difficulty_steps,
    select_open_difficulty_steps,
    tap_steps,
)
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def test_tap_steps_emit_active_input_then_neutral_settle() -> None:
    steps = tap_steps(MenuInput.START, hold_frames=2, settle_frames=3, phase="open")

    assert [step.menu_input for step in steps] == [
        MenuInput.START,
        MenuInput.NEUTRAL,
    ]
    assert [step.frames for step in steps] == [2, 3]
    assert [step.phase for step in steps] == ["open", "open:settle"]


def test_select_difficulty_only_opens_popup() -> None:
    steps = select_difficulty_steps(_race_setup(difficulty="novice"))
    active_inputs = [step.menu_input for step in steps if step.menu_input != MenuInput.NEUTRAL]

    assert active_inputs == [MenuInput.ACCEPT]
    assert steps[0].phase == "select_difficulty:open"
    assert steps[0].frames == MENU_TIMING.menu_hold_frames


def test_select_open_difficulty_moves_to_expert() -> None:
    steps = select_open_difficulty_steps(_race_setup(difficulty="expert"))
    active_inputs = [step.menu_input for step in steps if step.menu_input != MenuInput.NEUTRAL]

    assert active_inputs == [
        MenuInput.DOWN,
        MenuInput.DOWN,
        MenuInput.ACCEPT,
    ]


def test_continue_after_race_uses_menu_confirm_after_terminal_edge() -> None:
    active_inputs = [continue_after_race_step(index).menu_input for index in range(4)]

    assert active_inputs == [
        MenuInput.ACCEPT,
        MenuInput.ACCEPT,
        MenuInput.ACCEPT,
        MenuInput.ACCEPT,
    ]


def test_continue_next_course_uses_menu_confirm() -> None:
    step = continue_next_course_step()

    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase == "continue_after_race:next_course_accept"


def test_machine_select_moves_to_target_slot_before_selecting() -> None:
    steps = machine_select_steps(_race_setup(difficulty="novice", machine_select_column=1))
    active_inputs = [step.menu_input for step in steps if step.menu_input != MenuInput.NEUTRAL]

    assert active_inputs == [MenuInput.RIGHT]


def test_machine_select_default_slot_emits_no_directional_inputs() -> None:
    steps = machine_select_steps(_race_setup(difficulty="novice"))

    assert steps == ()


def test_menu_facts_normalize_native_menu_info() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "course_select",
            "game_mode_raw": 10,
            "menu_selected_mode_raw": 0,
            "menu_transition_state_raw": 3,
            "menu_current_ghost_type_raw": 0,
            "queued_game_mode_raw": 1,
            "course_index": 13,
            "difficulty_raw": 2,
            "camera_setting_name": "close_behind",
            "race_time_ms": 500,
            "episode_completion_fraction": 0.25,
            "race_laps_completed": 1,
            "total_lap_count": 3,
        }
    )

    assert facts.is_course_select
    assert not facts.in_gp_race
    assert facts.game_mode_raw == 10
    assert facts.selected_gp_mode
    assert facts.transition_state_raw == 3
    assert facts.current_ghost_type_raw == 0
    assert facts.queued_game_mode_raw == 1
    assert facts.course_select_cup_index == 2
    assert facts.difficulty_raw == 2
    assert facts.camera_setting == "close_behind"
    assert facts.race_time_ms == 500.0
    assert facts.completion_fraction == 0.25
    assert not facts.completed_race_laps
    assert not facts.terminal_race_result


def test_observed_menu_screen_keeps_open_difficulty_popup_owned() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "main_menu",
            "menu_selected_mode_raw": 5,
        }
    )

    assert (
        observed_menu_screen(
            facts,
            difficulty_popup_state=DifficultyPopupState.OPEN,
        )
        is ObservedMenuScreen.DIFFICULTY_POPUP
    )


def test_observed_menu_screen_accepts_popup_latch_only_on_gp_tile() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "main_menu",
            "menu_selected_mode_raw": 0,
        }
    )

    assert (
        observed_menu_screen(
            facts,
            difficulty_popup_state=DifficultyPopupState.OPEN,
        )
        is ObservedMenuScreen.DIFFICULTY_POPUP
    )


def test_menu_facts_detect_completed_laps_without_terminal_result() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "gp_race",
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
    )

    assert facts.in_gp_race
    assert facts.completed_race_laps
    assert not facts.terminal_race_result


def test_menu_facts_detect_terminal_race_result() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "gp_race",
            "entered_finished": True,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
    )

    assert facts.in_gp_race
    assert facts.terminal_race_result


def test_menu_facts_keep_sticky_finished_out_of_fsm_terminal_result() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "gp_race",
            "finished": True,
            "race_laps_completed": 0,
            "race_intro_timer": 30,
            "race_time_ms": 0,
            "total_lap_count": 3,
            "course_index": 0,
        }
    )

    assert facts.finished
    assert not facts.terminal_race_result
    assert facts.fresh_race_ready_for_policy


def test_menu_facts_detect_fresh_race_ready_for_policy() -> None:
    facts = MenuFacts.from_info(
        {
            "course_index": 0,
            "game_mode": "gp_race",
            "race_intro_timer": 30,
            "race_laps_completed": 0,
            "race_time_ms": 0,
            "total_lap_count": 3,
        }
    )

    assert facts.fresh_race_ready_for_policy


def test_menu_facts_reject_sparse_result_frame_as_fresh_race() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "gp_race",
            "race_intro_timer": 30,
            "race_time_ms": 0,
        }
    )

    assert not facts.fresh_race_ready_for_policy


def test_menu_facts_reject_unknown_course_as_fresh_race() -> None:
    facts = MenuFacts.from_info(
        {
            "game_mode": "gp_race",
            "race_intro_timer": 30,
            "race_laps_completed": 0,
            "race_time_ms": 0,
            "total_lap_count": 3,
        }
    )

    assert not facts.fresh_race_ready_for_policy


def test_menu_facts_detect_skippable_post_gp_screens() -> None:
    assert MenuFacts.from_info({"game_mode": "gp_end_cutscene"}).is_skippable_post_gp_screen
    assert MenuFacts.from_info({"game_mode": "skippable_credits"}).is_skippable_post_gp_screen


def test_raw_step_rejects_invalid_frame_count() -> None:
    with pytest.raises(ValueError, match="frames"):
        tap_steps(MenuInput.START, hold_frames=0, settle_frames=1, phase="invalid")


def _race_setup(
    *,
    difficulty: str,
    machine_select_column: int = 0,
) -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty=difficulty,
        cup_id="jack",
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=machine_select_column,
        engine_setting_id="balanced",
        engine_setting_raw_value=50,
    )
