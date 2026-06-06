# tests/ui/test_viewer_runtime.py
from pathlib import Path

import numpy as np
import pytest

from fzerox_emulator.arrays import Float32Array, UInt8Array
from rl_fzerox.core.career_mode.runner.controller import (
    CareerModeController,
    CareerRuntimeSession,
    _info_terminal_reason,
)
from rl_fzerox.core.career_mode.runner.menu import (
    MENU_TIMING,
    CareerPhase,
    MenuInput,
    RawMenuStep,
    camera_setting,
)
from rl_fzerox.core.career_mode.runner.race import SaveRaceExecutionPlan, SaveRaceSetup
from rl_fzerox.core.envs.engine.reset.camera import CAMERA_SYNC_CONTROLS
from rl_fzerox.core.envs.observations import (
    ImageStateObservation,
    observation_state,
    state_feature_names,
)
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    ObservationStateComponentConfig,
    StateFeatureDropoutGroupConfig,
    TrackRecordEntryConfig,
    TrackRecordsConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainConfig,
    WatchAppConfig,
)
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
    open_career_mode_runtime_session,
)
from rl_fzerox.ui.watch.runtime.career_mode.worker import (
    _active_policy_timing,
    _reset_race_progress_info,
    _set_session_control_timing,
    _snapshot_action_repeat,
    _snapshot_target_control_fps,
    _target_game_fps,
)
from rl_fzerox.ui.watch.runtime.episode import (
    _update_best_finish_position,
    _update_best_finish_ranks,
    _update_best_finish_times,
    _update_failed_track_attempts,
    _update_latest_finish_deltas_ms,
    _update_latest_finish_times,
)
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    configured_watch_zeroed_features,
)
from rl_fzerox.ui.watch.runtime.policy import _persist_reload_error
from rl_fzerox.ui.watch.runtime.session import WatchRuntimeSession
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _adjust_control_fps,
    _resolve_control_fps,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.screen.render import (
    _add_config_track_info,
    _observation_state_feature_names,
    _track_pool_records,
)
from tests.core.envs.helpers import CameraSyncBackend
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_watch_state_feature_zeroing_masks_selected_features_without_mutating_source() -> None:
    observation: ImageStateObservation = {
        "image": _sample_image(),
        "state": _sample_state([0.0, 2.0, 3.0]),
    }
    info = {
        "observation_state_features": (
            "vehicle_state.speed_kph_norm",
            "machine_context.energy_norm",
            "course_context.course_builtin_0",
        ),
        "observation_zeroed_state_features": ("vehicle_state.speed_kph_norm",),
    }

    masked_observation, masked_info = apply_watch_state_feature_zeroing(
        observation,
        info,
        watch_zeroed_features=frozenset({"machine_context.energy_norm"}),
    )

    assert observation["state"][1] == 2.0
    masked_state = observation_state(masked_observation)
    assert masked_state is not None
    assert masked_state[1] == 0.0
    assert masked_state[2] == 3.0
    assert masked_info["watch_zeroed_state_features"] == ("machine_context.energy_norm",)
    assert masked_info["observation_zeroed_state_features"] == (
        "machine_context.energy_norm",
        "vehicle_state.speed_kph_norm",
    )


def test_watch_state_feature_zeroing_supports_component_level_course_toggle() -> None:
    observation: ImageStateObservation = {
        "image": _sample_image(),
        "state": _sample_state([1.0, 0.0, 2.0]),
    }
    info = {
        "observation_state_features": (
            "course_context.course_builtin_0",
            "course_context.course_builtin_1",
            "vehicle_state.speed_kph_norm",
        ),
        "observation_zeroed_state_features": (),
    }

    masked_observation, masked_info = apply_watch_state_feature_zeroing(
        observation,
        info,
        watch_zeroed_features=frozenset({"course_context"}),
    )

    masked_state = observation_state(masked_observation)
    assert masked_state is not None
    assert list(masked_state) == [0.0, 0.0, 2.0]
    assert masked_info["watch_zeroed_state_features"] == ("course_context",)


def test_configured_watch_zeroed_features_inherits_dropout_one_groups(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    state_components = (
        ObservationStateComponentConfig(name="track_position", progress_source="segment_progress"),
        ObservationStateComponentConfig(name="course_context", encoding="one_hot_builtin"),
    )
    feature_names = state_feature_names(
        state_components=tuple(component.data() for component in state_components),
    )
    course_feature_names = tuple(
        feature_name for feature_name in feature_names if feature_name.startswith("course_context.")
    )
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                layout_continuous_axes=("steer",),
                layout_discrete_axes=("gas", "boost", "lean"),
            ),
            observation=ObservationConfig(mode="image_state", state_components=state_components),
        ),
        train=TrainConfig(
            state_feature_dropout_groups=(
                StateFeatureDropoutGroupConfig(
                    feature_names=("track_position.edge_ratio",),
                    dropout_prob=1.0,
                ),
                StateFeatureDropoutGroupConfig(
                    feature_names=course_feature_names,
                    dropout_prob=1.0,
                ),
                StateFeatureDropoutGroupConfig(
                    feature_names=("track_position.lap_progress",),
                    dropout_prob=0.6,
                ),
            )
        ),
    )

    assert configured_watch_zeroed_features(config) == frozenset(
        {"track_position.edge_ratio", "course_context"}
    )


def test_viewer_state_feature_names_fall_back_to_image_state_config(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    state_components = (
        ObservationStateComponentConfig(name="vehicle_state"),
        ObservationStateComponentConfig(name="control_history", controls=("boost",)),
    )
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                layout_continuous_axes=("steer",),
                layout_discrete_axes=("gas", "boost", "lean"),
            ),
            observation=ObservationConfig(
                mode="image_state",
                state_components=state_components,
            ),
        ),
    )

    assert _observation_state_feature_names(config, {}) == state_feature_names(
        state_components=tuple(component.data() for component in state_components),
        split_lean_history=False,
    )


def test_career_mode_session_renders_display_without_policy_crop(tmp_path: Path) -> None:
    class _Emulator:
        native_fps = 60.0

        def __init__(self) -> None:
            self.render_count = 0

        def render(self) -> UInt8Array:
            self.render_count += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    emulator = _Emulator()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(
                observation=ObservationConfig(
                    resolution={"mode": "custom", "height": 72, "width": 96},
                ),
            ),
        ),
        emulator=emulator,
        native_fps=60.0,
        native_control_fps=30.0,
        target_control_fps=30.0,
        target_control_seconds=1.0 / 30.0,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )

    session.render()

    assert emulator.render_count == 1


def test_career_mode_session_seeds_only_from_runtime_attempt_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    seeds: list[int] = []
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.Emulator",
        _Emulator,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.seed_process",
        seeds.append,
    )
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        seed=99,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )
    session = open_career_mode_runtime_session(config)
    session.close()
    seeded_config = config.model_copy(
        update={"watch": config.watch.model_copy(update={"attempt_seed": 1234})}
    )
    seeded_session = open_career_mode_runtime_session(seeded_config)
    seeded_session.close()

    assert seeds == [1234]


def test_career_mode_session_uses_native_menu_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.session.Emulator",
        _Emulator,
    )
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=4),
    )
    session = open_career_mode_runtime_session(config)
    session.close()

    assert session.native_control_fps == 60.0
    assert session.target_control_fps == 60.0
    assert session.target_control_seconds == pytest.approx(1.0 / 60.0)


def test_career_mode_menu_info_clears_policy_race_progress() -> None:
    info = _reset_race_progress_info(
        {
            "episode_step": 123,
            "episode_return": 456.0,
            "step_reward": 7.0,
            "progress_frontier_stalled_frames": 8,
            "stalled_steps": 9,
            "frames_run": 10,
            "repeat_index": 1,
            "reward_breakdown": {"progress": 1.0},
        }
    )

    assert info["episode_step"] == 0
    assert info["episode_return"] == 0.0
    assert info["step_reward"] == 0.0
    assert info["progress_frontier_stalled_frames"] == 0
    assert info["stalled_steps"] == 0
    assert info["frames_run"] == 0
    assert info["repeat_index"] == 0
    assert "reward_breakdown" not in info


def test_watch_fps_helpers_resolve_split_control_and_render_rates() -> None:
    assert _resolve_control_fps("auto", native_control_fps=30.0) == 30.0
    assert _resolve_control_fps("unlimited", native_control_fps=30.0) is None
    assert _resolve_control_fps(120.0, native_control_fps=30.0) == 120.0
    assert _resolve_render_fps(None, native_fps=60.0) == 60.0
    assert _resolve_render_fps("auto", native_fps=60.0) == 60.0
    assert _resolve_render_fps("unlimited", native_fps=60.0) is None


def test_watch_control_fps_adjustment_supports_uncapped_mode() -> None:
    assert _adjust_control_fps(60.0, 1, native_control_fps=60.0) == 65.0
    assert _adjust_control_fps(60.0, -1, native_control_fps=60.0) == 55.0
    assert _adjust_control_fps(None, 1, native_control_fps=60.0) is None
    assert _adjust_control_fps(None, -1, native_control_fps=60.0) == 55.0


def test_rate_meter_reset_discards_previous_phase_timing() -> None:
    meter = RateMeter(window=4)
    meter.tick(0.0)
    meter.tick(1.0)

    assert meter.rate_hz() == pytest.approx(1.0)

    meter.reset()

    assert meter.rate_hz() == 0.0
    meter.tick(10.0)
    meter.tick(10.5)
    assert meter.rate_hz() == pytest.approx(2.0)


def test_career_mode_terminal_reason_ignores_lap_counter_without_terminal_fact() -> None:
    reason = _info_terminal_reason(
        info={"race_laps_completed": 3, "total_lap_count": 3},
    )

    assert reason is None


def test_career_mode_lap_count_alone_does_not_finish_active_gp_race() -> None:
    reason = _info_terminal_reason(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )

    assert reason is None


def test_career_mode_active_gp_race_ignores_sticky_finish_flag() -> None:
    reason = _info_terminal_reason(
        info={
            "game_mode": "gp_race",
            "finished": True,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )

    assert reason is None


def test_career_mode_active_gp_race_uses_entered_finish_edge() -> None:
    reason = _info_terminal_reason(
        info={
            "game_mode": "gp_race",
            "entered_finished": True,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )

    assert reason == "finished"


def test_career_mode_camera_setting_reads_live_viewer_telemetry_key() -> None:
    assert camera_setting({"camera_setting_name": "close_behind"}) == "close_behind"
    assert camera_setting({"camera_setting": "regular"}) == "regular"


def test_career_mode_clears_active_policy_runner_outside_gp_race() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    session = object.__new__(WatchRuntimeSession)
    policy_control = object()

    active_control = controller.active_policy_control(
        session=session,
        current_policy_control=policy_control,
        info={"game_mode": "select_mode"},
    )

    assert active_control is None


def test_career_mode_does_not_expose_policy_before_policy_phase() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    session = object.__new__(WatchRuntimeSession)

    active_control = controller.active_policy_control(
        session=session,
        current_policy_control=None,
        info={"game_mode": "gp_race"},
    )

    assert active_control is None


def test_career_mode_enters_policy_phase_with_explicit_handoff_frame() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._resolve_policy_control = lambda info: object()
    controller._next_camera_sync_step = lambda info: None

    step = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 0,
            "total_lap_count": 3,
        }
    )

    assert step is not None
    assert step.phase == "policy_race:handoff"
    assert controller.policy_owns_control()


def test_career_mode_policy_phase_stays_policy_owned_during_race() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)

    step = controller.next_raw_step(info={"game_mode": "gp_race"})

    assert step is None
    assert controller.policy_owns_control()


def test_career_mode_policy_phase_rejects_menu_step_without_terminal_result() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)

    with pytest.raises(RuntimeError, match="before observing a game result"):
        controller.next_raw_step(info={"game_mode": "course_select"})


def test_career_mode_ignores_policy_adapter_truncation() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._attempt_id = "attempt-a"

    terminal_handled = controller.observe_step(
        session=_session_without_telemetry(),
        info={
            "game_mode": "gp_race",
            "termination_reason": "progress_stalled",
            "truncation_reason": "progress_stalled",
        },
    )

    assert terminal_handled is False
    assert controller.policy_owns_control()


def test_career_mode_observe_step_reports_unhandled_non_terminal_race() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._attempt_id = None

    terminal_handled = controller.observe_step(
        session=_session_without_telemetry(),
        info={"game_mode": "gp_race"},
    )

    assert terminal_handled is False
    assert controller.policy_owns_control()


def test_career_mode_observe_step_ignores_result_screen_without_terminal_fact() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._attempt_id = None

    terminal_handled = controller.observe_step(
        session=_session_without_telemetry(),
        info={"game_mode": "results", "race_laps_completed": 3, "total_lap_count": 3},
    )

    assert terminal_handled is False
    assert controller.policy_owns_control()


def test_career_mode_observe_step_handles_finished_race_screen_still_in_gp_mode() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._attempt_id = None

    terminal_handled = controller.observe_step(
        session=_session_without_telemetry(),
        info={
            "game_mode": "gp_race",
            "entered_finished": True,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )

    assert terminal_handled is True
    assert not controller.policy_owns_control()


def test_career_mode_observe_step_handles_post_gp_screen_as_finished() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._attempt_id = None

    terminal_handled = controller.observe_step(
        session=_session_without_telemetry(),
        info={"game_mode": "gp_end_cutscene"},
    )

    assert terminal_handled is True
    assert not controller.policy_owns_control()


def test_career_mode_continues_explicit_terminal_result_edge() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._resolve_policy_control = lambda info: object()
    controller._pending_steps.clear()
    controller._awaiting_new_race_after_terminal = True

    step = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "entered_finished": True,
            "race_laps_completed": 3,
            "race_time_ms": 105_000,
            "total_lap_count": 3,
        }
    )
    assert step is not None
    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase.startswith("continue_after_race:")
    assert not controller.policy_owns_control()


def test_career_mode_does_not_continue_lap_three_without_terminal_result() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._resolve_policy_control = lambda info: object()
    controller._pending_steps.clear()
    controller._awaiting_new_race_after_terminal = True
    controller._continuing_race_result = True

    step = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 3,
            "race_time_ms": 105_000,
            "total_lap_count": 3,
        }
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "continue_after_race:wait_for_fresh_race"


def test_career_mode_clears_stale_post_race_inputs_on_unknown_screen() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._pending_steps.append(
        RawMenuStep(menu_input=MenuInput.START, frames=2, phase="unsafe:start")
    )
    controller._awaiting_new_race_after_terminal = True
    controller._observed_terminal_race_result = True

    step = controller.next_raw_step(info={"game_mode": "results"})

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "continue_after_race:wait_for_known_screen"
    assert all(
        pending_step.menu_input is not MenuInput.START for pending_step in controller._pending_steps
    )


def test_career_mode_result_continuation_releases_accept_between_pulses() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._resolve_policy_control = lambda info: object()
    controller._pending_steps.clear()
    controller._awaiting_new_race_after_terminal = True
    controller._observed_terminal_race_result = True

    first_step = controller.next_raw_step(info={"game_mode": "gp_race", "entered_finished": True})
    release_step = controller.next_raw_step(info={"game_mode": "gp_race"})

    assert first_step is not None
    assert first_step.menu_input is MenuInput.ACCEPT
    assert release_step is not None
    assert release_step.menu_input is MenuInput.NEUTRAL
    assert release_step.phase.endswith(":settle")


def test_career_mode_post_race_never_drains_start_pulses_inside_gp_race() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._pending_steps.append(
        RawMenuStep(menu_input=MenuInput.START, frames=2, phase="unsafe:start")
    )
    controller._awaiting_new_race_after_terminal = True
    controller._continuing_race_result = False

    step = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 2,
            "race_time_ms": 95_000,
            "total_lap_count": 3,
        },
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "continue_after_race:wait_for_fresh_race"
    assert not controller._pending_steps


def test_career_mode_enter_race_waits_out_stale_post_result_gp_race() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._resolve_policy_control = lambda info: object()
    controller._awaiting_new_race_after_terminal = True
    controller._observed_terminal_race_result = True

    step = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 0,
            "race_intro_timer": 0,
            "race_time_ms": 9633,
            "termination_reason": "crashed",
            "total_lap_count": 3,
        },
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "enter_race:wait_for_fresh_race"
    assert not controller.policy_owns_control()


def test_career_mode_continues_gp_end_cutscene_without_press_budget() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._start_presses_in_phase = 12
    controller._pending_steps.clear()

    step = controller.next_raw_step(info={"game_mode": "gp_end_cutscene"})

    assert step is not None
    assert step.menu_input is MenuInput.A_BUTTON
    assert step.phase == "continue_after_race:post_gp_screen"
    assert controller._start_presses_in_phase == 0


def test_career_mode_continues_gp_next_course_screen_with_accept() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._start_presses_in_phase = 12
    controller._pending_steps.clear()

    step = controller.next_raw_step(info={"game_mode": "gp_race_next_course"})

    assert step is not None
    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase == "continue_after_race:next_course_accept"
    assert controller._start_presses_in_phase == 0


def test_career_mode_continues_skippable_credits_without_press_budget() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)
    controller._start_presses_in_phase = 12
    controller._pending_steps.clear()

    step = controller.next_raw_step(info={"game_mode": "skippable_credits"})

    assert step is not None
    assert step.menu_input is MenuInput.A_BUTTON
    assert step.phase == "continue_after_race:post_gp_screen"
    assert controller._start_presses_in_phase == 0


def test_career_mode_post_race_mode_select_restarts_target_selection() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)

    step = controller.next_raw_step(info={"game_mode": "main_menu"})

    assert step is not None
    assert step.phase == "select_difficulty:open"
    assert controller.phase is CareerPhase.SELECT_DIFFICULTY


def test_career_mode_active_policy_timing_preserves_speed_multiplier(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=2),
    )
    policy_config = config.model_copy(
        update={"env": config.env.model_copy(update={"action_repeat": 4})}
    )

    class _Session:
        native_fps = 60.0

        @staticmethod
        def snapshot_config(base_config: WatchAppConfig) -> WatchAppConfig:
            return policy_config

    timing = _active_policy_timing(
        config,
        _Session(),
        native_control_fps=30.0,
        target_control_fps=60.0,
    )

    assert timing.target_fps == 30.0
    assert timing.target_seconds == pytest.approx(1.0 / 30.0)


def test_career_mode_snapshot_target_control_fps_tracks_current_controller(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=2),
    )
    policy_config = config.model_copy(
        update={"env": config.env.model_copy(update={"action_repeat": 4})}
    )

    class _Session:
        native_fps = 60.0

        @staticmethod
        def snapshot_config(base_config: WatchAppConfig) -> WatchAppConfig:
            return policy_config

    assert (
        _snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=False,
        )
        == 60.0
    )
    assert (
        _snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=True,
        )
        == 15.0
    )


def test_career_mode_target_game_fps_uses_active_action_repeat() -> None:
    assert _target_game_fps(target_control_fps=15.0, action_repeat=4) == 60.0
    assert _target_game_fps(target_control_fps=None, action_repeat=4) is None


def test_career_mode_menu_snapshots_use_native_frame_repeat(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=4),
    )

    assert _snapshot_action_repeat(config, policy_active=False) == 1
    assert _snapshot_action_repeat(config, policy_active=True) == 4


def test_career_mode_session_timing_updates_with_viewer_commands() -> None:
    class _Session:
        target_control_fps: float | None = 30.0
        target_control_seconds: float | None = 1.0 / 30.0

    session = _Session()

    _set_session_control_timing(
        session,
        target_control_fps=60.0,
        target_control_seconds=1.0 / 60.0,
    )

    assert session.target_control_fps == 60.0
    assert session.target_control_seconds == pytest.approx(1.0 / 60.0)


def test_career_mode_menu_transition_does_not_skip_machine_settings() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.SELECT_MACHINE)

    step = controller._next_menu_step({"game_mode": "machine_select"})

    assert step is not None
    assert step.phase.startswith("enter_machine_settings:start:")


def test_career_mode_menu_transition_does_not_skip_race_entry() -> None:
    controller = _minimal_career_controller(
        phase=CareerPhase.APPLY_ENGINE,
        engine_applied=True,
    )

    step = controller._next_menu_step({"game_mode": "machine_settings"})

    assert step is not None
    assert step.phase.startswith("enter_race:start:")


def test_career_mode_rejects_non_default_engine_without_ram_patch() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.APPLY_ENGINE)
    controller._setup = SaveRaceSetup(
        difficulty="novice",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_id="70",
        engine_setting_raw_value=70,
    )

    with pytest.raises(RuntimeError, match="non-default engine"):
        controller.before_step(
            session=_StubCareerSession(),
            info={"game_mode": "machine_settings"},
        )


def test_career_mode_accepts_default_engine_without_ram_patch() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.APPLY_ENGINE)
    session = _StubCareerSession()

    controller.before_step(
        session=session,
        info={"game_mode": "machine_settings"},
    )

    assert controller._engine_applied
    assert not session.patch_calls


@pytest.mark.parametrize(
    ("phase", "mode", "expected_step_prefix"),
    (
        (CareerPhase.BOOT_TO_DIFFICULTY, "course_select", "select_cup:"),
        (CareerPhase.ENTER_COURSE_SELECT, "machine_select", "enter_machine_settings:"),
        (CareerPhase.SELECT_CUP, "machine_settings", "apply_engine:"),
        (CareerPhase.CONTINUE_AFTER_RACE, "machine_select", "enter_machine_settings:"),
    ),
)
def test_career_mode_menu_accepts_already_advanced_screens(
    phase: CareerPhase,
    mode: str,
    expected_step_prefix: str,
) -> None:
    controller = _minimal_career_controller(phase=phase)

    step = controller._next_menu_step({"game_mode": mode})

    assert step is not None
    assert step.phase.startswith(expected_step_prefix)


def test_career_mode_fsm_reaches_policy_handoff_from_menu_path() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.BOOT_TO_DIFFICULTY)
    controller._resolve_policy_control = lambda info: object()
    controller._next_camera_sync_step = lambda info: None

    assert _next_active_menu_phase(controller, {"game_mode": "main_menu"}) == (
        "select_difficulty:open"
    )
    assert _next_active_menu_phase(controller, {"game_mode": "main_menu"}) == (
        "select_difficulty:accept"
    )
    assert (
        _next_active_menu_phase(
            controller,
            {"game_mode": "course_select", "course_index": 0},
        )
        == "enter_machine_select:start:1"
    )
    assert _next_active_menu_phase(controller, {"game_mode": "machine_select"}) == (
        "enter_machine_settings:start:1"
    )
    assert (
        _next_non_settle_menu_phase(
            controller,
            {"game_mode": "machine_settings"},
        )
        == "apply_engine:wait"
    )
    controller._engine_applied = True
    assert _next_active_menu_phase(controller, {"game_mode": "machine_settings"}) == (
        "enter_race:start:1"
    )

    handoff = controller.next_raw_step(
        info={
            "game_mode": "gp_race",
            "race_laps_completed": 0,
            "race_time_ms": 0,
            "total_lap_count": 3,
        }
    )

    assert handoff is not None
    assert handoff.phase == "policy_race:handoff"
    assert controller.policy_owns_control()


def test_career_mode_uses_start_to_enter_course_select_from_gp_mode() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_COURSE_SELECT)

    step = _next_non_settle_menu_step(
        controller,
        {"game_mode": "main_menu", "menu_selected_mode_raw": 0},
    )

    assert step.menu_input is MenuInput.START
    assert step.phase == "enter_course_select:start:1"
    assert step.frames == MENU_TIMING.start_hold_frames


def test_career_mode_reselects_gp_mode_before_difficulty_popup() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_COURSE_SELECT)

    step = _next_non_settle_menu_step(
        controller,
        {"game_mode": "main_menu", "menu_selected_mode_raw": 2},
    )

    assert step.menu_input is MenuInput.START
    assert step.phase == "enter_course_select:start:1"
    assert step.frames == MENU_TIMING.start_hold_frames


def test_career_mode_waits_for_course_select_during_main_menu_transition() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_COURSE_SELECT)

    step = controller.next_raw_step(
        info={
            "game_mode": "main_menu",
            "game_mode_raw": 0x8007,
            "menu_selected_mode_raw": 0,
        }
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "enter_course_select:wait_for_transition"


def test_career_mode_selects_gp_mode_before_difficulty_popup() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.BOOT_TO_DIFFICULTY)

    step = _next_non_settle_menu_step(
        controller,
        {"game_mode": "main_menu", "menu_selected_mode_raw": 2},
    )

    assert step.menu_input is MenuInput.LEFT
    assert step.phase == "select_mode:left_to_gp"
    assert controller.phase is CareerPhase.BOOT_TO_DIFFICULTY

    step = _next_non_settle_menu_step(
        controller,
        {"game_mode": "main_menu", "menu_selected_mode_raw": 0},
    )

    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase == "select_difficulty:open"


def test_career_mode_returned_main_menu_selects_gp_before_retry() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.CONTINUE_AFTER_RACE)

    step = _next_non_settle_menu_step(
        controller,
        {"game_mode": "main_menu", "menu_selected_mode_raw": 3},
    )

    assert step.menu_input is MenuInput.LEFT
    assert step.phase == "select_mode:left_to_gp"
    assert controller.phase is CareerPhase.BOOT_TO_DIFFICULTY


def test_career_mode_apply_execution_plan_resumes_post_race_continuation() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.POLICY_RACE)
    controller._pending_steps.append(RawMenuStep(MenuInput.START, 1, "stale"))
    controller._save_game_id = "save-a"
    controller._store = _ExecutionPlanStore()
    controller._active_policy_key = ("old-run", "best")

    controller._apply_execution_plan(
        SaveRaceExecutionPlan(
            attempt_id="attempt-b",
            policy_run_id="run-b",
            policy_run_dir=Path("/tmp/run-b"),
            policy_artifact="best",
            policy_algorithm="maskable_ppo",
            policy_path=Path("/tmp/run-b/policy.zip"),
            race_setup=SaveRaceSetup(
                difficulty="standard",
                cup_id="queen",
                course_id=None,
                vehicle_id="blue_falcon",
                vehicle_display_name="Blue Falcon",
                character_index=0,
                machine_select_slot=0,
                machine_select_row=0,
                machine_select_column=0,
                engine_setting_id="50",
                engine_setting_raw_value=50,
            ),
        )
    )

    assert controller.phase is CareerPhase.CONTINUE_AFTER_RACE
    assert not controller._pending_steps
    assert controller._attempt_id == "attempt-b"
    assert controller._setup.difficulty == "standard"
    assert controller._setup.cup_id == "queen"
    assert controller._active_policy_key is None
    assert controller._awaiting_new_race_after_terminal
    assert controller._continuing_race_result


def test_career_mode_camera_sync_uses_normal_reset_camera_helper() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._camera_setting = "close_behind"
    controller._camera_synced = False
    session = _CameraSyncCareerSession()
    info = {
        "game_mode": "gp_race",
        "camera_setting_name": "regular",
        "menu_transition_state_raw": 0,
        "race_intro_timer": CAMERA_SYNC_CONTROLS.ready_intro_timer - 1,
        "race_time_ms": 83,
    }

    advanced = controller.before_step(session=session, info=info)

    assert advanced is True
    assert controller._camera_synced is True
    assert info["camera_setting"] == "close_behind"
    assert info["camera_setting_sync"] == "changed"
    assert info["camera_setting_taps"] == 3


def test_career_mode_camera_sync_waits_until_intro_allows_camera_change() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._camera_setting = "close_behind"
    controller._camera_synced = False

    step = controller._next_camera_sync_step(
        {
            "camera_setting_name": "regular",
            "race_intro_timer": CAMERA_SYNC_CONTROLS.ready_intro_timer + 1,
            "race_time_ms": 1,
        }
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "camera_sync:wait_intro"


def test_career_mode_camera_sync_waits_while_menu_transition_is_active() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._camera_setting = "close_behind"
    controller._camera_synced = False

    step = controller._next_camera_sync_step(
        {
            "camera_setting_name": "regular",
            "menu_transition_state_raw": 3,
            "race_intro_timer": CAMERA_SYNC_CONTROLS.ready_intro_timer - 1,
            "race_time_ms": 0,
        }
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "camera_sync:wait_intro"


def test_career_mode_camera_sync_step_yields_to_runtime_sync() -> None:
    controller = _minimal_career_controller(phase=CareerPhase.ENTER_RACE)
    controller._camera_setting = "close_behind"
    controller._camera_synced = False

    step = controller._next_camera_sync_step(
        {
            "camera_setting_name": "regular",
            "menu_transition_state_raw": 0,
            "race_intro_timer": CAMERA_SYNC_CONTROLS.ready_intro_timer - 1,
            "race_time_ms": 83,
        }
    )

    assert step is not None
    assert step.menu_input is MenuInput.NEUTRAL
    assert step.phase == "camera_sync:apply"


def test_best_finish_position_tracks_only_finished_episodes() -> None:
    best_position = _update_best_finish_position(
        None,
        {"termination_reason": "crashed", "position": 4},
        _sample_telemetry(position=4),
    )
    assert best_position is None

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=8),
    )
    assert best_position == 8

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=12),
    )
    assert best_position == 8

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=3),
    )
    assert best_position == 3


def test_best_finish_times_track_successful_finishes_per_track() -> None:
    best_times = _update_best_finish_times(
        {},
        {"termination_reason": "crashed", "race_time_ms": 98_000, "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000),
    )
    assert best_times == {}

    best_times = _update_best_finish_times(
        best_times,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000),
    )
    best_times = _update_best_finish_times(
        best_times,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=101_000),
    )
    best_times = _update_best_finish_times(
        best_times,
        {"termination_reason": "finished", "track_id": "silence"},
        _sample_telemetry(race_time_ms=105_000),
    )
    best_times = _update_best_finish_times(
        best_times,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=95_000),
    )

    assert best_times == {"mute": 95_000, "silence": 105_000}


def test_best_finish_ranks_track_successful_finishes_per_track() -> None:
    best_ranks = _update_best_finish_ranks(
        {},
        {"termination_reason": "crashed", "position": 1, "track_id": "mute"},
        _sample_telemetry(position=1),
    )
    assert best_ranks == {}

    best_ranks = _update_best_finish_ranks(
        best_ranks,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(position=8),
    )
    best_ranks = _update_best_finish_ranks(
        best_ranks,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(position=12),
    )
    best_ranks = _update_best_finish_ranks(
        best_ranks,
        {"termination_reason": "finished", "track_id": "silence"},
        _sample_telemetry(position=5),
    )
    best_ranks = _update_best_finish_ranks(
        best_ranks,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(position=3),
    )

    assert best_ranks == {"mute": 3, "silence": 5}


def test_latest_finish_times_track_most_recent_successful_finish_per_track() -> None:
    latest_times = _update_latest_finish_times(
        {},
        {"termination_reason": "crashed", "race_time_ms": 98_000, "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000),
    )
    assert latest_times == {}

    latest_times = _update_latest_finish_times(
        latest_times,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000),
    )
    latest_times = _update_latest_finish_times(
        latest_times,
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=101_000),
    )
    latest_times = _update_latest_finish_times(
        latest_times,
        {"termination_reason": "finished", "track_id": "silence"},
        _sample_telemetry(race_time_ms=105_000),
    )

    assert latest_times == {"mute": 101_000, "silence": 105_000}


def test_latest_finish_delta_tracks_previous_pb_gap() -> None:
    latest_deltas = _update_latest_finish_deltas_ms(
        {},
        {},
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000),
    )
    assert latest_deltas == {}

    latest_deltas = _update_latest_finish_deltas_ms(
        latest_deltas,
        {"mute": 98_000},
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=101_000),
    )
    assert latest_deltas == {"mute": 3_000}

    latest_deltas = _update_latest_finish_deltas_ms(
        latest_deltas,
        {"mute": 98_000},
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=95_000),
    )
    assert latest_deltas == {"mute": -3_000}


def test_failed_track_attempts_track_until_success() -> None:
    failed_attempts = _update_failed_track_attempts(
        frozenset(),
        {"truncation_reason": "progress_stalled", "track_id": "mute"},
        episode_done=True,
    )
    assert failed_attempts == frozenset({"mute"})

    failed_attempts = _update_failed_track_attempts(
        failed_attempts,
        {"termination_reason": "crashed", "track_id": "silence"},
        episode_done=False,
    )
    assert failed_attempts == frozenset({"mute"})

    failed_attempts = _update_failed_track_attempts(
        failed_attempts,
        {"termination_reason": "finished", "track_id": "mute"},
        episode_done=True,
    )
    assert failed_attempts == frozenset()


def test_record_panel_marks_failed_watch_attempts_until_success() -> None:
    records: tuple[dict[str, object], ...] = (
        {
            "track_id": "mute",
            "track_course_id": "mute_city",
            "track_course_name": "Mute City",
        },
    )

    failed_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        best_finish_ranks={},
        best_finish_times={},
        latest_finish_times={},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset({"mute"}),
    )[0]
    success_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        best_finish_ranks={},
        best_finish_times={"mute": 95_000},
        latest_finish_times={"mute": 95_000},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset({"mute"}),
    )[0]

    assert failed_section.lines[0].status_text == "FAILED"
    assert failed_section.lines[2].value == "failed"
    assert success_section.lines[0].status_text == ""
    assert success_section.lines[2].value == "1:35.000"


def test_config_track_info_uses_registry_name_for_course_index(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    baseline_path = tmp_path / "mute.state"
    core_path.touch()
    rom_path.touch()
    baseline_path.write_bytes(b"baseline")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        display_name="Mute City Time Attack - Blue Falcon Balanced",
                        baseline_state_path=baseline_path,
                        course_index=0,
                    ),
                ),
            )
        ),
    )
    info: dict[str, object] = {"course_index": 0}

    _add_config_track_info(info, config)

    assert info["track_id"] == "mute_city"
    assert info["track_display_name"] == "Mute City Time Attack - Blue Falcon Balanced"


def test_config_track_info_uses_active_curriculum_track_pool(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    mute_baseline_path = tmp_path / "mute.state"
    port_baseline_path = tmp_path / "port.state"
    white_land_baseline_path = tmp_path / "white_land.state"
    core_path.touch()
    rom_path.touch()
    mute_baseline_path.write_bytes(b"mute")
    port_baseline_path.write_bytes(b"port")
    white_land_baseline_path.write_bytes(b"white")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        display_name="Mute City Time Attack - Blue Falcon Balanced",
                        baseline_state_path=mute_baseline_path,
                        course_index=0,
                    ),
                ),
            )
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(name="jack"),
                CurriculumStageConfig(
                    name="queen_seed",
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="port_town",
                                display_name="Port Town Time Attack - Blue Falcon Balanced",
                                baseline_state_path=port_baseline_path,
                                course_index=7,
                                records=TrackRecordsConfig(
                                    non_agg_best=TrackRecordEntryConfig(time_ms=73_000),
                                ),
                            ),
                            TrackSamplingEntryConfig(
                                id="white_land",
                                display_name="White Land Time Attack - Blue Falcon Balanced",
                                baseline_state_path=white_land_baseline_path,
                                course_index=8,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    info: dict[str, object] = {"course_index": 7, "curriculum_stage": 1}

    _add_config_track_info(info, config)

    assert [record["track_id"] for record in _track_pool_records(config, info)] == [
        "port_town",
        "white_land",
    ]
    assert info["track_id"] == "port_town"
    assert info["track_display_name"] == "Port Town Time Attack - Blue Falcon Balanced"
    assert info["track_non_agg_best_time_ms"] == 73_000


def test_track_sampling_records_prefer_refreshed_watch_snapshot_state(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    old_baseline_path = tmp_path / "old.state"
    new_baseline_path = tmp_path / "new.state"
    core_path.touch()
    rom_path.touch()
    old_baseline_path.write_bytes(b"old")
    new_baseline_path.write_bytes(b"new")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="old",
                        course_id="x_cup_old",
                        runtime_course_key="x_cup_slot_1",
                        baseline_state_path=old_baseline_path,
                    ),
                ),
            )
        ),
    )
    refreshed = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="new",
                course_id="x_cup_new",
                runtime_course_key="x_cup_slot_1",
                baseline_state_path=new_baseline_path,
            ),
        ),
    )

    records = _track_pool_records(config, active_track_sampling=refreshed)

    assert records[0]["track_id"] == "new"
    assert records[0]["track_course_id"] == "x_cup_new"
    assert records[0]["track_reset_course_key"] == "x_cup_slot_1"


def test_record_rows_click_stable_runtime_course_key() -> None:
    section = track_record_sections(
        current_info={},
        track_pool_records=(
            {
                "track_id": "generated",
                "track_course_id": "x_cup_generated",
                "track_reset_course_key": "x_cup_slot_1",
            },
        ),
        best_finish_ranks={},
        best_finish_times={},
        latest_finish_times={},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset(),
    )[0]

    assert section.lines[0].click_course_id == "x_cup_slot_1"


def test_persist_reload_error_writes_full_message_once(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "watch" / "runtime"
    runtime_dir.mkdir(parents=True)

    logged_error = _persist_reload_error(
        reload_error="PyTorchStreamReader failed reading file data/0",
        runtime_dir=runtime_dir,
        last_logged_reload_error=None,
    )

    assert logged_error == "PyTorchStreamReader failed reading file data/0"
    assert (tmp_path / "watch" / "reload_error.log").read_text(encoding="utf-8") == (
        "PyTorchStreamReader failed reading file data/0\n"
    )


def _sample_image() -> UInt8Array:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def _sample_state(values: list[float]) -> Float32Array:
    return np.asarray(values, dtype=np.float32)


class _ExecutionPlanStore:
    def list_save_course_setups(self, save_game_id: str):
        assert save_game_id == "save-a"
        return ()

    def save_game_unlock_progress(self, save_game_id: str):
        assert save_game_id == "save-a"
        return None


def _minimal_career_controller(
    *,
    phase: CareerPhase,
    engine_applied: bool = False,
) -> CareerModeController:
    from collections import deque

    controller = object.__new__(CareerModeController)
    controller._phase = phase
    controller._pending_steps = deque()
    controller._start_presses_in_phase = 0
    controller._engine_applied = engine_applied
    controller._setup = SaveRaceSetup(
        difficulty="novice",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_id="50",
        engine_setting_raw_value=50,
    )
    controller._camera_setting = None
    controller._camera_synced = True
    controller._camera_sync_taps = 0
    controller._awaiting_new_race_after_terminal = False
    controller._continuing_race_result = False
    controller._observed_terminal_race_result = False
    controller._continue_after_race_pulses = 0
    controller._fresh_race_ready_frames = 0
    return controller


def _next_active_menu_phase(
    controller: CareerModeController,
    info: dict[str, object],
) -> str:
    for _ in range(16):
        step = _next_non_settle_menu_step(controller, info)
        if step.menu_input.value != "neutral":
            return step.phase
    raise AssertionError("Career Mode FSM emitted only neutral steps")


def _next_non_settle_menu_phase(
    controller: CareerModeController,
    info: dict[str, object],
) -> str:
    return _next_non_settle_menu_step(controller, info).phase


def _next_non_settle_menu_step(
    controller: CareerModeController,
    info: dict[str, object],
):
    for _ in range(16):
        step = controller.next_raw_step(info=info)
        assert step is not None
        if not step.phase.endswith(":settle"):
            return step
    raise AssertionError("Career Mode FSM emitted only settle steps")


def _session_without_telemetry() -> CareerRuntimeSession:
    class _Emulator:
        @staticmethod
        def try_read_telemetry() -> None:
            return None

    class _Session:
        emulator = _Emulator()

    return _Session()


class _CameraSyncCareerSession:
    def __init__(self) -> None:
        self.emulator = CameraSyncBackend(camera_setting_raw=2)


class _StubCareerSession:
    def __init__(self) -> None:
        self.patch_calls: list[tuple[str, int]] = []
        self.emulator = self

    def try_read_telemetry(self) -> None:
        return None

    def patch_engine_settings(
        self,
        *,
        mode: str,
        engine_setting_raw_value: int,
    ) -> None:
        self.patch_calls.append((mode, engine_setting_raw_value))
