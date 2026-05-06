# tests/ui/test_viewer_runtime.py
from pathlib import Path

import numpy as np

from fzerox_emulator.arrays import Float32Array, UInt8Array
from rl_fzerox.core.config.schema import (
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
from rl_fzerox.core.envs.observations import (
    ImageStateObservation,
    observation_state,
    state_feature_names,
)
from rl_fzerox.ui.watch.runtime.episode import (
    _update_best_finish_position,
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
from rl_fzerox.ui.watch.runtime.timing import (
    _adjust_control_fps,
    _resolve_control_fps,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.screen.render import _add_config_track_info, _track_pool_records
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
        feature_name
        for feature_name in feature_names
        if feature_name.startswith("course_context.")
    )
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                name="configured_hybrid",
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
        best_finish_times={},
        latest_finish_times={},
        latest_finish_deltas_ms={},
        failed_track_attempts=frozenset({"mute"}),
    )[0]
    success_section = track_record_sections(
        current_info={},
        track_pool_records=records,
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
