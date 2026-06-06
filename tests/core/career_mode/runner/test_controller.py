# tests/core/career_mode/runner/test_controller.py

from __future__ import annotations

from rl_fzerox.core.career_mode.runner.controller import _info_terminal_reason


def test_terminal_reason_ignores_lap_counter_without_terminal_flag() -> None:
    reason = _info_terminal_reason(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 3,
            "total_lap_count": 3,
            "finished": False,
            "retired": False,
            "crashed": False,
        }
    )

    assert reason is None


def test_terminal_reason_ignores_sticky_native_terminal_flags() -> None:
    assert _info_terminal_reason(info={"finished": True}) is None
    assert _info_terminal_reason(info={"retired": True}) is None
    assert _info_terminal_reason(info={"crashed": True}) is None


def test_terminal_reason_uses_native_entered_state_edges() -> None:
    assert _info_terminal_reason(info={"entered_finished": True}) == "finished"
    assert _info_terminal_reason(info={"entered_retired": True}) == "retired"
    assert _info_terminal_reason(info={"entered_crashed": True}) == "crashed"


def test_terminal_reason_uses_native_reason() -> None:
    assert _info_terminal_reason(info={"termination_reason": "finished"}) == "finished"


def test_terminal_reason_uses_post_gp_screen() -> None:
    assert _info_terminal_reason(info={"game_mode": "gp_end_cutscene"}) == "finished"
    assert _info_terminal_reason(info={"game_mode": "skippable_credits"}) == "finished"


def test_terminal_reason_ignores_training_failure_reasons() -> None:
    assert _info_terminal_reason(info={"termination_reason": "spinning_out"}) is None
    assert _info_terminal_reason(info={"termination_reason": "falling_off_track"}) is None
