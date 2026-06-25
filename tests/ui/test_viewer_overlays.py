# tests/ui/test_viewer_overlays.py
"""Watch viewer tests for game-overlay labels and runtime rates.

These cases lock down the compact labels drawn over the game preview, including
course identity, save notices, and displayed game-speed multipliers.
"""

from rl_fzerox.ui.watch.view.screen.frame import (
    _game_course_overlay_label,
    _game_speed_overlay_label,
    _game_status_overlay_label,
)
from rl_fzerox.ui.watch.view.screen.view_model import _with_viewer_rates


def test_game_course_overlay_label_prefers_cup_and_course_name() -> None:
    assert (
        _game_course_overlay_label(
            {
                "track_course_ref": "joker/big_hand",
                "track_course_name": "Big Hand",
            }
        )
        == "Joker Cup : Big Hand"
    )


def test_game_course_overlay_label_marks_locked_course_lightly() -> None:
    assert (
        _game_course_overlay_label(
            {
                "track_course_id": "big_hand",
                "track_course_ref": "joker/big_hand",
                "track_course_name": "Big Hand",
            },
            reset_info={"track_sampling_locked_course_id": "big_hand"},
        )
        == "> Joker Cup : Big Hand <"
    )


def test_game_course_overlay_label_uses_career_target_during_post_gp() -> None:
    assert (
        _game_course_overlay_label(
            {
                "career_mode_fsm_observed_screen": "post_gp",
                "career_mode_target_label": "Clear Master Joker Cup",
                "course_index": 55,
                "game_mode": "unskippable_credits",
            }
        )
        == "Clear Master Joker Cup"
    )


def test_game_speed_overlay_label_formats_actual_speedup() -> None:
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 120.0},
            action_repeat=2,
        )
        == "2.0x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 150.0},
            action_repeat=2,
        )
        == "2.5x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "control_fps": 45.0},
            action_repeat=2,
        )
        == "1.5x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 720.0},
            action_repeat=2,
        )
        == "12.0x"
    )


def test_game_status_overlay_label_uses_watch_save_notice() -> None:
    assert _game_status_overlay_label({"watch_save_notice": "alt baseline saved"}) == (
        "alt baseline saved"
    )
    assert _game_status_overlay_label({"watch_save_notice": "   "}) is None


def test_viewer_rates_preserve_runtime_game_fps() -> None:
    info = _with_viewer_rates(
        {"game_fps": 72.0},
        native_fps=60.0,
        action_repeat=2,
        current_control_fps=60.0,
        current_render_fps=60.0,
        target_control_fps=60.0,
        target_render_fps=60.0,
    )

    assert info["game_fps"] == 72.0
    assert _game_speed_overlay_label(info, action_repeat=2) == "1.2x"
