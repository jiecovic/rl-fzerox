# tests/ui/test_viewer_game_panel_controls.py
from rl_fzerox.core.envs.actions import RACE_CONTROL_MASKS
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.panels.visuals.viz import _control_viz
from tests.ui.viewer_game_panel_support import (
    _race_section,
    race_control_state,
)
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_dash_pad_boost_lights_single_boost_pill() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(state_labels=("active", "dash_pad_boost"), boost_timer=12),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels
    assert "dash" not in active_labels


def test_manual_boost_timer_lights_boost_pill() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(boost_timer=12),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels


def test_signed_boost_timer_lights_generic_boost_pill() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(boost_timer=-1),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels


def test_control_viz_snaps_discrete_gas_to_button_state() -> None:
    viz = _control_viz(
        race_control_state(control_mask=RACE_CONTROL_MASKS.accelerate),
        gas_level=0.37,
        continuous_drive_enabled=False,
        force_full_throttle=False,
    )

    assert viz.gas_level == 1.0
