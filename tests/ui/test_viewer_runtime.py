# tests/ui/test_viewer_runtime.py
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fzerox_emulator.arrays import Float32Array, UInt8Array
from rl_fzerox.core.domain.observation_image import CustomResolutionChoice
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
from rl_fzerox.ui.watch.records import track_record_key
from rl_fzerox.ui.watch.runtime.career_mode.menu import reset_race_progress_info
from rl_fzerox.ui.watch.runtime.career_mode.session import (
    CareerModeRuntimeSession,
    open_career_mode_runtime_session,
)
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    active_policy_timing,
    set_session_control_timing,
    snapshot_action_repeat,
    snapshot_target_control_fps,
    target_game_fps,
)
from rl_fzerox.ui.watch.runtime.episode import (
    _update_best_finish_position,
    _update_best_finish_rank_setups,
    _update_best_finish_rank_times,
    _update_best_finish_ranks,
    _update_best_finish_time_ranks,
    _update_best_finish_time_setups,
    _update_best_finish_times,
    _update_failed_track_attempts,
    _update_latest_finish_deltas_ms,
    _update_latest_finish_times,
    _update_track_attempt_stats,
)
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    configured_watch_zeroed_features,
)
from rl_fzerox.ui.watch.runtime.policy import _persist_reload_error
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
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_watch_state_feature_zeroing_masks_selected_features_without_mutating_source() -> None:
    observation: ImageStateObservation = {
        "image": _sample_image(),
        "state": _sample_state([0.0, 2.0, 3.0]),
    }
    info: dict[str, object] = {
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
    emulator: Any = _Emulator()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(
                observation=ObservationConfig(
                    resolution=CustomResolutionChoice(height=72, width=96),
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
    info = reset_race_progress_info(
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

    timing = active_policy_timing(
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
        snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=False,
        )
        == 60.0
    )
    assert (
        snapshot_target_control_fps(
            config=config,
            session=_Session(),
            native_control_fps=60.0,
            target_control_fps=60.0,
            policy_active=True,
        )
        == 15.0
    )


def test_career_mode_target_game_fps_uses_active_action_repeat() -> None:
    assert target_game_fps(target_control_fps=15.0, action_repeat=4) == 60.0
    assert target_game_fps(target_control_fps=None, action_repeat=4) is None


def test_career_mode_menu_snapshots_use_native_frame_repeat(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action_repeat=4),
    )

    assert snapshot_action_repeat(config, policy_active=False) == 1
    assert snapshot_action_repeat(config, policy_active=True) == 4


def test_career_mode_session_timing_updates_with_viewer_commands() -> None:
    class _Session:
        target_control_fps: float | None = 30.0
        target_control_seconds: float | None = 1.0 / 30.0

    session = _Session()

    set_session_control_timing(
        session,
        target_control_fps=60.0,
        target_control_seconds=1.0 / 60.0,
    )

    assert session.target_control_fps == 60.0
    assert session.target_control_seconds == pytest.approx(1.0 / 60.0)


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


def test_best_finish_time_metadata_tracks_the_finish_that_set_the_time() -> None:
    best_times: dict[str, int] = {}
    time_ranks: dict[str, int] = {}
    time_setups: dict[str, dict[str, str | int]] = {}
    info = {
        "termination_reason": "finished",
        "track_id": "mute",
        "race_time_ms": 98_000,
        "position": 3,
        "track_vehicle_name": "Deep Claw",
        "track_engine_setting_raw_value": 60,
    }

    time_ranks = _update_best_finish_time_ranks(
        time_ranks,
        best_times,
        info,
        _sample_telemetry(race_time_ms=98_000, position=3),
    )
    time_setups = _update_best_finish_time_setups(
        time_setups,
        best_times,
        info,
        None,
    )
    best_times = _update_best_finish_times(
        best_times,
        info,
        _sample_telemetry(race_time_ms=98_000),
    )
    slower_info: dict[str, object] = dict(
        info,
        race_time_ms=101_000,
        position=1,
        track_vehicle_name="Blue Falcon",
        track_engine_setting_raw_value=40,
    )
    time_ranks = _update_best_finish_time_ranks(
        time_ranks,
        best_times,
        slower_info,
        _sample_telemetry(race_time_ms=101_000, position=1),
    )
    time_setups = _update_best_finish_time_setups(time_setups, best_times, slower_info, None)

    assert best_times == {"mute": 98_000}
    assert time_ranks == {"mute": 3}
    assert time_setups == {
        "mute": {
            "vehicle_name": "Deep Claw",
            "engine_setting_raw_value": 60,
        }
    }


def test_best_finish_times_track_gp_difficulties_separately() -> None:
    novice_info: dict[str, object] = {
        "termination_reason": "finished",
        "track_id": "mute",
        "track_mode": "gp_race",
        "track_gp_difficulty": "novice",
    }
    expert_info: dict[str, object] = {
        "termination_reason": "finished",
        "track_id": "mute",
        "track_mode": "gp_race",
        "track_gp_difficulty": "expert",
    }
    novice_key = track_record_key(novice_info)
    expert_key = track_record_key(expert_info)
    assert novice_key is not None
    assert expert_key is not None

    best_times = _update_best_finish_times(
        {},
        novice_info,
        _sample_telemetry(race_time_ms=98_000),
    )
    best_times = _update_best_finish_times(
        best_times,
        expert_info,
        _sample_telemetry(race_time_ms=101_000),
    )
    best_times = _update_best_finish_times(
        best_times,
        novice_info,
        _sample_telemetry(race_time_ms=95_000),
    )

    assert best_times == {novice_key: 95_000, expert_key: 101_000}


def test_x_cup_record_key_uses_generated_hash_and_difficulty() -> None:
    assert (
        track_record_key(
            {
                "track_course_id": "x_cup_slot_1",
                "track_generated_course_kind": "x_cup",
                "track_generated_course_hash": "abcd1234",
                "track_gp_difficulty": "expert",
            }
        )
        == "x_cup:abcd1234#difficulty=expert"
    )


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


def test_best_finish_rank_metadata_tracks_time_for_best_position() -> None:
    best_ranks: dict[str, int] = {}
    rank_times: dict[str, int] = {}
    rank_setups: dict[str, dict[str, str | int]] = {}
    info: dict[str, object] = {
        "termination_reason": "finished",
        "track_id": "mute",
        "race_time_ms": 101_000,
        "position": 1,
        "track_vehicle_name": "Blue Falcon",
        "track_engine_setting_raw_value": 50,
    }

    rank_setups = _update_best_finish_rank_setups(
        rank_setups,
        rank_times,
        best_ranks,
        info,
        None,
    )
    rank_times = _update_best_finish_rank_times(
        rank_times,
        best_ranks,
        info,
        _sample_telemetry(race_time_ms=101_000, position=1),
    )
    best_ranks = _update_best_finish_ranks(
        best_ranks,
        info,
        _sample_telemetry(position=1),
    )
    faster_same_rank_info: dict[str, object] = dict(
        info,
        race_time_ms=98_000,
        track_vehicle_name="Deep Claw",
        track_engine_setting_raw_value=60,
    )
    rank_setups = _update_best_finish_rank_setups(
        rank_setups,
        rank_times,
        best_ranks,
        faster_same_rank_info,
        None,
    )
    rank_times = _update_best_finish_rank_times(
        rank_times,
        best_ranks,
        faster_same_rank_info,
        _sample_telemetry(race_time_ms=98_000, position=1),
    )

    assert best_ranks == {"mute": 1}
    assert rank_times == {"mute": 98_000}
    assert rank_setups == {
        "mute": {
            "vehicle_name": "Deep Claw",
            "engine_setting_raw_value": 60,
        }
    }


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


def test_track_attempt_stats_track_finish_rate_and_average_completion() -> None:
    stats = _update_track_attempt_stats(
        {},
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.25,
        },
        None,
        episode_done=True,
    )
    stats = _update_track_attempt_stats(
        stats,
        {
            "termination_reason": "finished",
            "track_id": "mute",
            "episode_completion_fraction": 1.0,
        },
        None,
        episode_done=True,
    )
    stats = _update_track_attempt_stats(
        stats,
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.5,
        },
        None,
        episode_done=False,
    )

    assert stats == {
        "mute": {
            "attempts": 2,
            "finishes": 1,
            "completion_samples": 2,
            "completion_sum": 1.25,
            "best_completion": 1.0,
        }
    }


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
        best_finish_rank_times={},
        best_finish_rank_setups={},
        best_finish_times={},
        best_finish_time_ranks={},
        best_finish_time_setups={},
        latest_finish_times={},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset({"mute"}),
    )[0]
    success_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        best_finish_ranks={},
        best_finish_rank_times={},
        best_finish_rank_setups={},
        best_finish_times={"mute": 95_000},
        best_finish_time_ranks={},
        best_finish_time_setups={},
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
                        display_name="Mute City Time Attack - Blue Falcon Engine 50",
                        baseline_state_path=baseline_path,
                        course_index=0,
                    ),
                ),
            )
        ),
    )
    info: dict[str, object] = {"course_index": 0}

    _add_config_track_info(info, config)

    assert info["track_entry_id"] == "mute_city"
    assert info["track_id"] == "mute_city"
    assert info["track_course_key"] == "course:0"
    assert info["track_display_name"] == "Mute City Time Attack - Blue Falcon Engine 50"


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
                        display_name="Mute City Time Attack - Blue Falcon Engine 50",
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
                                display_name="Port Town Time Attack - Blue Falcon Engine 50",
                                baseline_state_path=port_baseline_path,
                                course_index=7,
                                records=TrackRecordsConfig(
                                    non_agg_best=TrackRecordEntryConfig(time_ms=73_000),
                                ),
                            ),
                            TrackSamplingEntryConfig(
                                id="white_land",
                                display_name="White Land Time Attack - Blue Falcon Engine 50",
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

    assert [record["track_entry_id"] for record in _track_pool_records(config, info)] == [
        "port_town",
        "white_land",
    ]
    assert [record["track_course_key"] for record in _track_pool_records(config, info)] == [
        "course:7",
        "course:8",
    ]
    assert info["track_entry_id"] == "port_town"
    assert info["track_id"] == "port_town"
    assert info["track_course_key"] == "course:7"
    assert info["track_display_name"] == "Port Town Time Attack - Blue Falcon Engine 50"
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
                mode="gp_race",
                course_index=48,
                generated_course_kind="x_cup",
                generated_course_seed=123,
                generated_course_hash="newhash",
                generated_course_slot=0,
                generated_course_generation=2,
            ),
        ),
    )

    records = _track_pool_records(config, active_track_sampling=refreshed)

    assert records[0]["track_entry_id"] == "new"
    assert records[0]["track_id"] == "new"
    assert records[0]["track_course_key"] == "x_cup_slot_1"
    assert records[0]["track_course_id"] == "x_cup_new"
    assert records[0]["track_reset_course_key"] == "x_cup_slot_1"
    assert records[0]["track_generated_course_kind"] == "x_cup"
    assert records[0]["track_generated_course_generation"] == 2


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
        best_finish_rank_times={},
        best_finish_rank_setups={},
        best_finish_times={},
        best_finish_time_ranks={},
        best_finish_time_setups={},
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
