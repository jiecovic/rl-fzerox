# tests/ui/test_viewer_panel_sections.py
"""Watch viewer tests for side-panel section content.

The cases verify run state, episode progress, policy output, runtime, training,
and auxiliary prediction sections without mixing in overlay or state-vector
layout details.
"""

from pathlib import Path

import numpy as np

from fzerox_emulator import RaceControlState
from rl_fzerox.core.runtime_spec.schema import (
    PolicyConfig,
    TrainConfig,
)
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from tests.ui.viewer_support import record_book
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_side_panel_drops_cockpit_control_section() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    assert "Cockpit Control" not in [section.title for section in columns.left]
    assert [section.title for section in columns.left] == [
        "Run State",
        "Episode Progress",
        "Policy Output",
        "Race State",
        "Race Setup",
        "Runtime",
    ]
    assert [section.title for section in columns.middle] == [
        "Track Geometry",
    ]


def test_train_tab_shows_current_training_hparams() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        emulator_renderer="gliden64",
        watch_device="cpu",
        train_config=TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            device="cuda",
            total_timesteps=50_000_000,
            learning_rate=2e-4,
            n_steps=2048,
            output_root=Path("local/runs"),
            run_name="wide_ppo",
        ),
        policy_config=PolicyConfig.model_validate({"extractor": {"conv_profile": "nature"}}),
    )

    live_policy_section = next(
        section for section in columns.left if section.title == "Policy Output"
    )
    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    training_section = next(section for section in columns.train if section.title == "Training")
    policy_section = next(section for section in columns.train if section.title == "Policy")
    live_policy_values = {line.label: line.value for line in live_policy_section.lines}
    runtime_values = {line.label: line.value for line in runtime_section.lines}
    training_values = {line.label: line.value for line in training_section.lines}
    policy_values = {line.label: line.value for line in policy_section.lines}

    assert training_values["Algorithm"] == "maskable_hybrid_action_ppo"
    assert training_values["Train device"] == "cuda"
    assert "Renderer" not in training_values
    assert "Device" not in training_values
    assert "Run" not in training_values
    assert training_values["Target steps"] == "50,000,000"
    assert training_values["LR"] == "2e-04"
    assert live_policy_values["Device"] == "cpu"
    assert runtime_values["Renderer"] == "gliden64"
    assert policy_values["Conv"] == "nature"
    assert policy_values["Pi net"] == "[256, 256]"


def test_state_tab_shows_auxiliary_state_predictions_next_to_targets() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        policy_config=PolicyConfig.model_validate(
            {
                "auxiliary_state": {
                    "enabled": True,
                    "losses": [
                        {
                            "name": "track_position.edge_ratio",
                            "weight": 0.5,
                            "grounded_only": True,
                        },
                        {
                            "name": "surface_state.on_refill_surface",
                            "weight": 0.25,
                            "grounded_only": False,
                        },
                        {
                            "name": "course_context.builtin_course_id",
                            "weight": 1.0,
                            "grounded_only": False,
                        },
                    ],
                }
            }
        ),
        policy_auxiliary_state_predictions={
            "track_position.edge_ratio": 0.125,
            "surface_state.on_refill_surface": 0.26,
            "course_context.builtin_course_id": {"index": 7, "confidence": 0.82},
        },
        policy_auxiliary_state_targets={
            "track_position.edge_ratio": 0.25,
            "surface_state.on_refill_surface": 0.0,
            "course_context.builtin_course_id": {"index": 3},
        },
    )

    state_section = next(section for section in columns.stats if section.title == "State Vector")
    edge_ratio_line = next(
        line for line in state_section.lines if line.label == "edge_ratio (ground)"
    )
    refill_line = next(line for line in state_section.lines if line.label == "on_refill_surface")
    course_line = next(line for line in state_section.lines if line.label == "course id")

    assert edge_ratio_line.value == " 0.1250 |  0.2500 |    6% |        "
    assert refill_line.value == " 0.2600 |  0.0000 |  hit  |        "
    assert course_line.value == " 7@82% |       3 | miss  |        "


def test_session_section_shows_episode_step_counter() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "episode_step": 123,
            "reverse_timer": 45,
            "progress_frontier_stalled_frames": 300,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        max_episode_steps=50_000,
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
    episode_frame_line = next(
        line for line in session_section.lines if line.label == "Episode frame"
    )
    env_step_line = next(line for line in session_section.lines if line.label == "Env step")
    frontier_line = next(line for line in session_section.lines if line.label == "Frontier frames")
    assert episode_frame_line.value == "123 / 50000"
    assert env_step_line.value == "41 / 16667"
    assert frontier_line.value == "300 / 900"


def test_session_section_omits_reverse_and_stuck_counters() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "reverse_timer": 45,
            "stalled_steps": 17,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        max_episode_steps=50_000,
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
    labels = {line.label for line in session_section.lines}
    assert "Reverse frames" not in labels
    assert "Stuck frames" not in labels


def test_timing_section_shows_compact_runtime_rates() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "control_fps": 30.0,
            "game_fps": 60.0,
            "control_fps_target": 120.0,
            "game_fps_target": 240.0,
            "render_fps": 60.0,
            "render_fps_target": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    values = {line.label: line.value for line in runtime_section.lines}

    assert values == {
        "Renderer": "unknown",
        "Repeat": "x2",
        "Control FPS": "30.0",
        "Game FPS": "60.0",
        "Render FPS": "60.0",
        "Speed multiplier": "1.00x",
        "Game size": "444x592",
        "Obs size": "84x116x12",
    }


def test_timing_section_shows_game_speed_multiplier() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "control_fps": 60.0,
            "control_fps_target": 60.0,
            "render_fps": 60.0,
            "render_fps_target": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    speed_line = next(line for line in runtime_section.lines if line.label == "Speed multiplier")

    assert speed_line.value == "2.00x"


def test_session_section_shows_checkpoint_experience_from_frames() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_num_timesteps=660_000,
        policy_experience_frames=1_320_000,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")
    experience_line = next(line for line in session_section.lines if line.label == "Experience")

    assert experience_line.value == "6h 06m"


def test_session_section_keeps_mixed_action_repeat_lineage_experience() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_num_timesteps=10_000,
        policy_experience_frames=15_000,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")
    experience_line = next(line for line in session_section.lines if line.label == "Experience")

    assert experience_line.value == "4m"


def test_session_section_shows_policy_deterministic_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_deterministic=False,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Policy Output")
    deterministic_line = next(line for line in session_section.lines if line.label == "Mode")

    assert deterministic_line.value == "stochastic"


def test_session_section_shows_manual_driver_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_label="latest",
        policy_deterministic=False,
        manual_control_enabled=True,
        policy_action=None,
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")
    driver_line = next(line for line in session_section.lines if line.label == "Driver")

    assert driver_line.value == "manual"


def test_session_section_formats_hybrid_action_value_with_fixed_digits() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_deterministic=False,
        policy_action={
            "continuous": np.array([0.0, 0.5, -0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Policy Output")
    action_line = next(line for line in session_section.lines if line.label == "Action")

    assert action_line.value == "c=[+0.00,+0.50,-0.50] d=[0,0]"


def test_session_section_shows_reward_with_four_decimals() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "step_reward": -0.016,
        },
        reset_info={},
        episode_reward=-12.34567,
        paused=False,
        control_state=RaceControlState(),
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    episode_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
    step_line = next(line for line in episode_section.lines if line.label == "Step reward")
    return_line = next(line for line in episode_section.lines if line.label == "Return")

    assert step_line.value == "-0.0160"
    assert return_line.value == "-12.3457"


def test_session_section_omits_global_best_finish_position() -> None:
    columns = _build_panel_columns(
        episode=2,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        track_record_book=record_book(best_finish_position=8),
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")

    assert all(line.label != "Best position" for line in session_section.lines)
