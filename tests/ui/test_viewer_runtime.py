# tests/ui/test_viewer_runtime.py
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fzerox_emulator.arrays import Float32Array, UInt8Array
from rl_fzerox.core.career_mode.policy import CareerModePolicyControl
from rl_fzerox.core.domain.observation_image import CustomResolutionChoice
from rl_fzerox.core.envs.observations import (
    ImageStateObservation,
    observation_state,
    state_feature_names,
)
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    CareerModeRaceSetupConfig,
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
    WatchConfig,
)
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.ui.watch.records import TrackRecordBook, track_record_key
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    _career_attempt_game_rng_seed,
)
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
    _add_career_mode_info,
    _add_config_track_info,
    _observation_state_feature_names,
    _track_pool_records,
)
from tests.support.fakes import SyntheticBackend
from tests.ui.viewer_support import record_book, record_entry
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


def test_career_session_inherits_policy_dropout_one_zeroed_features(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_feature_dropouts = (
        config.observation.state_feature_dropouts[0].model_copy(update={"dropout_prob": 1.0}),
        config.observation.state_feature_dropouts[1].model_copy(update={"dropout_prob": 0.5}),
    )
    run = ManagedRun(
        id="policy-run",
        name="Policy Run",
        status="finished",
        config=config,
        config_hash="hash",
        run_dir=tmp_path / "policy-run",
        created_at="2026-01-01T00:00:00Z",
        lineage_id="policy-run",
    )
    policy_path = run.run_dir / "checkpoints" / "latest.zip"
    policy_path.parent.mkdir(parents=True)
    policy_path.touch()
    policy_control = CareerModePolicyControl(
        course_setup=ManagedSaveCourseSetup(
            id="course-setup",
            save_game_id="save",
            policy_run_id=run.id,
            policy_artifact="latest",
            engine_setting_raw_value=50,
            difficulty="novice",
            cup_id="jack",
            course_id="mute_city",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        policy_run=run,
        runner=PolicyRunner(
            LoadedPolicy(run_dir=run.run_dir, policy_path=policy_path, artifact="latest"),
            policy=_PolicyStub(),
        ),
    )
    emulator: Any = SyntheticBackend()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path)),
        emulator=emulator,
        native_fps=60.0,
        native_sample_rate=48_000.0,
        native_control_fps=60.0,
        target_control_fps=None,
        target_control_seconds=None,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )

    session.begin_policy_race(policy_control=policy_control, seed=7, course_id="mute_city")

    expected_zeroed_features = configured_watch_zeroed_features(
        session.snapshot_config(session.config)
    )
    assert session.watch_zeroed_state_features == expected_zeroed_features
    assert "track_position.edge_ratio" in expected_zeroed_features
    assert "track_position.outside_track_bounds" not in expected_zeroed_features


class _PolicyStub:
    def predict(
        self,
        observation: object,
        state: object | None = None,
        episode_start: object | None = None,
        deterministic: bool = True,
    ) -> tuple[int, object | None]:
        _ = (observation, episode_start, deterministic)
        return 0, state


def test_career_mode_render_info_keeps_runtime_target_and_attempt(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    info: dict[str, object] = {
        "career_mode_target_label": "Clear Expert Jack Cup",
        "career_mode_attempt_id": "live-attempt",
    }
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        watch=WatchConfig(
            manager_db_path=db_path,
            managed_save_game_id="save",
            save_attempt_id="launch-attempt",
            unlock_target_label="Clear Novice Joker Cup",
        ),
    )

    _add_career_mode_info(info, config)

    assert info["career_mode_target_label"] == "Clear Expert Jack Cup"
    assert info["career_mode_attempt_id"] == "live-attempt"


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
        native_sample_rate = 48_000

        def __init__(self) -> None:
            self.render_count = 0
            self.render_display_count = 0

        def render(self) -> UInt8Array:
            self.render_count += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def render_display(
            self,
            *,
            preset: str | None = None,
            height: int | None = None,
            width: int | None = None,
        ) -> UInt8Array:
            _ = (preset, height, width)
            self.render_display_count += 1
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
        native_sample_rate=48_000.0,
        native_control_fps=30.0,
        target_control_fps=30.0,
        target_control_seconds=1.0 / 30.0,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )

    session.render()

    assert emulator.render_count == 0
    assert emulator.render_display_count == 1


def test_career_mode_session_seeds_only_from_runtime_attempt_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0
        native_sample_rate = 48_000

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


def test_career_attempt_game_rng_seed_is_stable_per_attempt() -> None:
    first = _career_attempt_game_rng_seed(base_seed=1234, attempt_id="attempt-1")
    repeated = _career_attempt_game_rng_seed(base_seed=1234, attempt_id="attempt-1")
    next_attempt = _career_attempt_game_rng_seed(base_seed=1234, attempt_id="attempt-2")

    assert first == repeated
    assert first != next_attempt
    assert _career_attempt_game_rng_seed(base_seed=None, attempt_id="attempt-1") is None
    assert _career_attempt_game_rng_seed(base_seed=1234, attempt_id=None) is None


def test_career_mode_session_uses_native_menu_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Emulator:
        native_fps = 60.0
        native_sample_rate = 48_000

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


def test_track_record_book_tracks_successful_finishes_per_track() -> None:
    book = TrackRecordBook()
    book = book.update(
        {"termination_reason": "crashed", "race_time_ms": 98_000, "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000, position=4),
        episode_done=True,
    )
    assert book.best_finish_position is None
    assert book.entries["mute"].failed_attempt is True

    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000, position=8),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=101_000, position=12),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "silence"},
        _sample_telemetry(race_time_ms=105_000, position=5),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=95_000, position=3),
        episode_done=True,
    )

    assert book.best_finish_position == 3
    assert book.entries["mute"].best_finish_time_ms == 95_000
    assert book.entries["mute"].best_finish_time_rank == 3
    assert book.entries["mute"].best_finish_rank == 3
    assert book.entries["mute"].best_finish_rank_time_ms == 95_000
    assert book.entries["mute"].latest_finish_time_ms == 95_000
    assert book.entries["mute"].latest_finish_delta_ms == -3_000
    assert book.entries["mute"].failed_attempt is False
    assert book.entries["silence"].best_finish_time_ms == 105_000


def test_track_record_book_metadata_tracks_the_finish_that_set_the_record() -> None:
    book = TrackRecordBook()
    info = {
        "termination_reason": "finished",
        "track_id": "mute",
        "race_time_ms": 98_000,
        "position": 3,
        "track_vehicle_name": "Deep Claw",
        "track_engine_setting_raw_value": 60,
    }

    book = book.update(info, None, episode_done=True)
    slower_info: dict[str, object] = dict(
        info,
        race_time_ms=101_000,
        position=1,
        track_vehicle_name="Blue Falcon",
        track_engine_setting_raw_value=40,
    )
    book = book.update(
        slower_info,
        None,
        episode_done=True,
    )
    faster_same_rank_info: dict[str, object] = dict(
        info,
        race_time_ms=97_000,
        position=1,
        track_vehicle_name="Twin Noritta",
        track_engine_setting_raw_value=70,
    )
    book = book.update(
        faster_same_rank_info,
        None,
        episode_done=True,
    )

    entry = book.entries["mute"]
    assert entry.best_finish_time_ms == 97_000
    assert entry.best_finish_time_rank == 1
    assert entry.best_finish_time_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }
    assert entry.best_finish_rank == 1
    assert entry.best_finish_rank_time_ms == 97_000
    assert entry.best_finish_rank_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }
    assert entry.latest_finish_rank == 1
    assert entry.latest_finish_delta_ms == -1_000
    assert entry.latest_finish_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }


def test_track_record_book_tracks_gp_difficulties_separately() -> None:
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

    book = TrackRecordBook()
    book = book.update(novice_info, _sample_telemetry(race_time_ms=98_000), episode_done=True)
    book = book.update(expert_info, _sample_telemetry(race_time_ms=101_000), episode_done=True)
    book = book.update(novice_info, _sample_telemetry(race_time_ms=95_000), episode_done=True)

    assert book.entries[novice_key].best_finish_time_ms == 95_000
    assert book.entries[expert_key].best_finish_time_ms == 101_000


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


def test_track_record_book_tracks_failed_attempts_and_attempt_stats() -> None:
    book = TrackRecordBook()
    book = book.update(
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.25,
        },
        None,
        episode_done=True,
    )
    book = book.update(
        {
            "termination_reason": "finished",
            "track_id": "mute",
            "episode_completion_fraction": 1.0,
        },
        None,
        episode_done=True,
    )
    book = book.update(
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.5,
        },
        None,
        episode_done=False,
    )

    entry = book.entries["mute"]
    assert entry.failed_attempt is False
    assert entry.attempt_stats.as_mapping() == {
        "attempts": 2,
        "finishes": 1,
        "completion_samples": 2,
        "completion_sum": 1.25,
        "best_completion": 1.0,
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
        track_record_book=record_book({"mute": record_entry(failed_attempt=True)}),
    )[0]
    success_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        track_record_book=record_book(
            {
                "mute": record_entry(
                    best_finish_time_ms=95_000,
                    latest_finish_time_ms=95_000,
                    failed_attempt=True,
                )
            }
        ),
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


def test_track_sampling_records_count_in_memory_alt_baselines(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    base_path = tmp_path / "base.state"
    alt_a_path = tmp_path / "alt-a.state"
    alt_b_path = tmp_path / "alt-b.state"
    core_path.touch()
    rom_path.touch()
    base_path.write_bytes(b"base")
    alt_a_path.write_bytes(b"alt-a")
    alt_b_path.write_bytes(b"alt-b")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        display_name="Mute City",
                        course_id="mute_city",
                        baseline_state_path=base_path,
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_city__alt_alt-a",
                        display_name="Mute City alt A",
                        course_id="mute_city",
                        baseline_state_path=alt_a_path,
                        alt_baseline_id="alt-a",
                        alt_baseline_label="frame 100",
                        alt_baseline_source_entry_id="mute_city",
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_city__alt_alt-b",
                        display_name="Mute City alt B",
                        course_id="mute_city",
                        baseline_state_path=alt_b_path,
                        alt_baseline_id="alt-b",
                        alt_baseline_label="frame 200",
                        alt_baseline_source_entry_id="mute_city",
                    ),
                ),
            )
        ),
    )

    records = _track_pool_records(config)

    assert records[0]["track_alt_baseline_count"] == 2
    assert records[1]["track_alt_baseline_count"] == 0
    assert records[1]["track_alt_baseline_id"] == "alt-a"


def test_career_mode_track_pool_records_cover_selected_cup(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        watch=WatchConfig(
            managed_save_game_id="save-a",
            career_mode_race_setup=CareerModeRaceSetupConfig(
                difficulty="master",
                cup_id="joker",
                vehicle_id="blue_falcon",
                vehicle_display_name="Blue Falcon",
                character_index=0,
                machine_select_slot=0,
                machine_select_row=0,
                machine_select_column=0,
                engine_setting_raw_value=80,
            ),
        ),
    )

    records = _track_pool_records(config)

    assert [record["track_course_name"] for record in records] == [
        "Rainbow Road",
        "Devil's Forest 3",
        "Space Plant",
        "Sand Ocean 2",
        "Port Town 2",
        "Big Hand",
    ]
    assert {record["track_gp_difficulty"] for record in records} == {"master"}
    assert {record["track_vehicle_name"] for record in records} == {"Blue Falcon"}
    assert {record["track_engine_setting_raw_value"] for record in records} == {80}
    best_time = records[0]["track_non_agg_best_time_ms"]
    assert isinstance(best_time, int)
    assert best_time > 0


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
        track_record_book=record_book(),
    )[0]

    assert section.lines[0].click_course_id == "x_cup_slot_1"


def test_record_rows_can_disable_course_jump_clicks() -> None:
    section = track_record_sections(
        current_info={},
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_id": "mute_city",
                "track_reset_course_key": "mute_city",
            },
        ),
        track_record_book=record_book(),
        allow_course_jumps=False,
    )[0]

    assert section.lines[0].click_course_id is None


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
