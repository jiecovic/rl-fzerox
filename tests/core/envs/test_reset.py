# tests/core/envs/test_reset.py
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.engine.reset import TrackBaselineCache
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    EnvConfig,
    ObservationConfig,
    ObservationStateComponentConfig,
    TrackRecordEntryConfig,
    TrackRecordsConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from tests.core.envs.helpers import (
    CameraSyncAfterIntroBackend,
    CameraSyncBackend,
    ScriptedStepBackend,
    image_obs,
    telemetry,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_telemetry


def test_reset_returns_stacked_observation():
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=2,
            observation=ObservationConfig(frame_stack=4),
        ),
    )

    obs, info = env.reset(seed=123)
    obs = image_obs(obs)

    assert obs.shape == (84, 84, 12)
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert info["observation_shape"] == (84, 84, 12)
    assert info["observation_frame_shape"] == (84, 84, 3)
    assert info["observation_stack"] == 4
    assert np.array_equal(obs[:, :, 0:3], obs[:, :, 3:6])
    assert np.array_equal(obs[:, :, 3:6], obs[:, :, 6:9])
    assert np.array_equal(obs[:, :, 6:9], obs[:, :, 9:12])
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 2, 2, 3]


def test_reset_can_return_image_state_observation() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne", "can_boost"),
            speed_kph=750.0,
            energy=89.0,
            max_energy=178.0,
            reverse_timer=1,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            observation=ObservationConfig(
                mode="image_state",
                frame_stack=4,
                state_components=(ObservationStateComponentConfig(name="vehicle_state"),),
            ),
        ),
    )

    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (84, 84, 12)
    assert obs["image"].dtype == np.uint8
    assert obs["state"].shape == (8,)
    assert obs["state"].dtype == np.float32
    assert obs["state"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert info["observation_mode"] == "image_state"
    assert info["observation_shape"] == (84, 84, 12)
    assert info["observation_state_shape"] == (8,)
    assert info["observation_state_features"] == (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.boost_unlocked",
        "vehicle_state.boost_active",
        "vehicle_state.lateral_velocity_norm",
        "vehicle_state.sliding_active",
    )


def test_reset_info_is_pickle_safe_with_live_telemetry() -> None:
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1),
    )

    _, info = env.reset(seed=123)

    assert "telemetry" not in info
    pickle.dumps(info)


def test_reset_loads_sampled_track_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "silence.state"
    baseline_path.write_bytes(b"state")
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_path,
                        weight=1.0,
                        course_index=1,
                        records=TrackRecordsConfig(
                            non_agg_best=TrackRecordEntryConfig(
                                time_ms=60638,
                                player="Daniel",
                                date="2012-04-10",
                                mode="PAL",
                            ),
                            non_agg_worst=TrackRecordEntryConfig(time_ms=63279),
                        ),
                    ),
                ),
            ),
        ),
    )

    _, info = env.reset(seed=123)

    assert backend.loaded_baselines == []
    assert backend.loaded_baseline_bytes == [(baseline_path, len(b"state"))]
    assert info["track_sampling_enabled"] is True
    assert info["track_id"] == "silence"
    assert info["track_baseline_state_path"] == str(baseline_path)
    assert info["track_sampling_weight"] == 1.0
    assert info["track_course_index"] == 1
    assert info["track_non_agg_best_time_ms"] == 60638
    assert info["track_non_agg_best_player"] == "Daniel"
    assert info["track_non_agg_best_mode"] == "PAL"
    assert info["track_non_agg_worst_time_ms"] == 63279
    assert info["baseline_kind"] == "custom"
    assert info["baseline_state_path"] == str(baseline_path)


def test_step_info_keeps_sampled_track_metadata(tmp_path: Path) -> None:
    baseline_path = tmp_path / "silence.state"
    baseline_path.write_bytes(b"state")
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        display_name="Silence Time Attack - Blue Falcon Balanced",
                        baseline_state_path=baseline_path,
                        weight=1.0,
                        course_index=1,
                        records=TrackRecordsConfig(
                            non_agg_best=TrackRecordEntryConfig(time_ms=60638),
                            non_agg_worst=TrackRecordEntryConfig(time_ms=63279),
                        ),
                    ),
                ),
            ),
        ),
    )
    env.reset(seed=123)

    _, _, _, _, step_info = env.step(env.action_space.sample())

    assert step_info["track_id"] == "silence"
    assert step_info["track_display_name"] == "Silence Time Attack - Blue Falcon Balanced"
    assert step_info["track_course_index"] == 1
    assert step_info["track_non_agg_worst_time_ms"] == 63279


def test_reset_can_disable_sampled_track_baseline_cache(tmp_path: Path) -> None:
    baseline_path = tmp_path / "silence.state"
    baseline_path.write_bytes(b"state")
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            cache_track_baselines=False,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_path,
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )

    env.reset(seed=123)

    assert backend.loaded_baselines == [baseline_path]
    assert backend.loaded_baseline_bytes == []


def test_terminal_reset_reuses_sampled_baseline_instead_of_continuing_race(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "silence.state"
    baseline_path.write_bytes(b"state")
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            reset_to_race=True,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_path,
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )
    env.reset(seed=123)
    env._engine._episode.done = True
    _, info = env.reset(seed=124)

    assert info["baseline_kind"] == "custom"
    assert info["track_id"] == "silence"
    assert backend.loaded_baseline_bytes == [
        (baseline_path, len(b"state")),
        (baseline_path, len(b"state")),
    ]


def test_track_baseline_cache_reads_each_state_once(tmp_path: Path) -> None:
    baseline_path = tmp_path / "silence.state"
    baseline_path.write_bytes(b"one")
    backend = SyntheticBackend()
    cache = TrackBaselineCache()

    cache.load_into_backend(backend, baseline_path)
    baseline_path.write_bytes(b"longer replacement")
    cache.load_into_backend(backend, baseline_path)

    assert backend.loaded_baseline_bytes == [
        (baseline_path, len(b"one")),
        (baseline_path, len(b"one")),
    ]


def test_track_baseline_cache_evicts_oldest_states_when_byte_limit_is_hit(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.state"
    second_path = tmp_path / "second.state"
    first_path.write_bytes(b"aaa")
    second_path.write_bytes(b"bbbb")
    backend = SyntheticBackend()
    cache = TrackBaselineCache(max_cached_state_bytes=6)

    cache.load_into_backend(backend, first_path)
    cache.load_into_backend(backend, second_path)
    first_path.write_bytes(b"ccccc")
    cache.load_into_backend(backend, first_path)

    assert backend.loaded_baseline_bytes == [
        (first_path, len(b"aaa")),
        (second_path, len(b"bbbb")),
        (first_path, len(b"ccccc")),
    ]


def test_track_baseline_cache_skips_caching_oversized_states(tmp_path: Path) -> None:
    baseline_path = tmp_path / "oversized.state"
    baseline_path.write_bytes(b"12345")
    backend = SyntheticBackend()
    cache = TrackBaselineCache(max_cached_state_bytes=4)

    cache.load_into_backend(backend, baseline_path)
    baseline_path.write_bytes(b"12")
    cache.load_into_backend(backend, baseline_path)

    assert backend.loaded_baseline_bytes == [
        (baseline_path, len(b"12345")),
        (baseline_path, len(b"12")),
    ]


def test_balanced_track_sampling_cycles_with_env_index_offset(tmp_path: Path) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence", "sand", "forest"))
    config = EnvConfig(
        action_repeat=1,
        track_sampling=TrackSamplingConfig(
            enabled=True,
            sampling_mode="balanced",
            entries=tuple(
                TrackSamplingEntryConfig(id=track_id, baseline_state_path=baseline_path)
                for track_id, baseline_path in baseline_paths.items()
            ),
        ),
    )
    env_zero = FZeroXEnv(backend=SyntheticBackend(), config=config, env_index=0)
    env_two = FZeroXEnv(backend=SyntheticBackend(), config=config, env_index=2)

    env_zero_ids = [env_zero.reset(seed=123)[1]["track_id"] for _ in range(5)]
    _, env_two_info = env_two.reset(seed=123)

    assert env_zero_ids == ["mute", "silence", "sand", "forest", "mute"]
    assert env_two_info["track_id"] == "sand"
    assert env_two_info["track_sampling_mode"] == "balanced"
    assert env_two_info["track_sampling_cycle_position"] == 2


def test_balanced_track_sampling_respects_weights(tmp_path: Path) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute",
                        baseline_state_path=baseline_paths["mute"],
                        weight=2.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_paths["silence"],
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )

    sampled_ids = [env.reset(seed=123)[1]["track_id"] for _ in range(6)]

    assert sampled_ids == ["mute", "silence", "mute", "mute", "silence", "mute"]


def test_sequential_track_sampling_uses_config_order_for_watch(tmp_path: Path) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence", "sand"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute",
                        baseline_state_path=baseline_paths["mute"],
                        weight=3.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_paths["silence"],
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="sand",
                        baseline_state_path=baseline_paths["sand"],
                        weight=2.0,
                    ),
                ),
            ),
        ),
        env_index=2,
    )

    env.set_sequential_track_sampling(True)
    sampled_infos = [env.reset(seed=123)[1] for _ in range(4)]

    assert [info["track_id"] for info in sampled_infos] == [
        "mute",
        "silence",
        "sand",
        "mute",
    ]
    assert sampled_infos[0]["track_sampling_mode"] == "sequential"
    assert sampled_infos[2]["track_sampling_cycle_position"] == 2


def test_sequential_track_sampling_rotates_courses_not_raw_pool_entries(
    tmp_path: Path,
) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute_a", "mute_b", "silence"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_a",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute_a"],
                        weight=1.0,
                        vehicle="blue_falcon",
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_b",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute_b"],
                        weight=5.0,
                        vehicle="fire_stingray",
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
                        baseline_state_path=baseline_paths["silence"],
                        weight=1.0,
                        vehicle="white_cat",
                    ),
                ),
            ),
        ),
    )

    env.set_sequential_track_sampling(True)
    first_info = env.reset(seed=123)[1]
    env.set_track_sampling_weights({"mute_a": 9.0, "mute_b": 1.0, "silence": 3.0})
    second_info = env.reset(seed=124)[1]
    third_info = env.reset(seed=125)[1]

    assert first_info["track_course_id"] == "mute_city"
    assert second_info["track_course_id"] == "silence"
    assert third_info["track_course_id"] == "mute_city"


def test_sequential_track_sampling_can_jump_to_requested_course_for_watch(
    tmp_path: Path,
) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence", "sand"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute"],
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
                        baseline_state_path=baseline_paths["silence"],
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="sand",
                        course_id="sand_ocean",
                        baseline_state_path=baseline_paths["sand"],
                        weight=1.0,
                    ),
                ),
            ),
        ),
    )

    env.set_sequential_track_sampling(True)
    first_info = env.reset(seed=123)[1]
    env.set_next_sequential_reset_course("sand_ocean")
    second_info = env.reset(seed=124)[1]
    third_info = env.reset(seed=125)[1]

    assert first_info["track_course_id"] == "mute_city"
    assert second_info["track_course_id"] == "sand_ocean"
    assert third_info["track_course_id"] == "mute_city"
    assert first_info["track_sampling_mode"] == "sequential"


@pytest.mark.parametrize("sampling_mode", ("step_balanced", "adaptive_step_balanced"))
def test_dynamic_track_sampling_accepts_runtime_weight_updates(
    tmp_path: Path,
    sampling_mode: Literal["step_balanced", "adaptive_step_balanced"],
) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode=sampling_mode,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute",
                        baseline_state_path=baseline_paths["mute"],
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        baseline_state_path=baseline_paths["silence"],
                    ),
                ),
            ),
        ),
    )

    first_info = env.reset(seed=123)[1]

    assert first_info["track_id"] in {"mute", "silence"}
    assert first_info["track_sampling_mode"] == sampling_mode

    env.set_track_sampling_weights({"mute": 0.0, "silence": 1.0})
    sampled_ids = [env.reset(seed=123)[1]["track_id"] for _ in range(4)]

    assert sampled_ids == ["silence", "silence", "silence", "silence"]
    assert env.reset(seed=123)[1]["track_sampling_mode"] == sampling_mode


def test_step_balanced_track_sampling_excludes_zero_weighted_courses(
    tmp_path: Path,
) -> None:
    baseline_paths = _write_track_baselines(tmp_path, ("mute", "silence", "sand"))
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute"],
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
                        baseline_state_path=baseline_paths["silence"],
                    ),
                    TrackSamplingEntryConfig(
                        id="sand",
                        course_id="sand_ocean",
                        baseline_state_path=baseline_paths["sand"],
                    ),
                ),
            ),
        ),
    )

    env.set_track_sampling_weights({"mute": 0.0, "silence": 0.0, "sand": 1.0})
    sampled_ids = [env.reset(seed=123)[1]["track_id"] for _ in range(4)]

    assert sampled_ids == ["sand", "sand", "sand", "sand"]


def test_step_balanced_track_sampling_zeroes_duplicate_course_entries(
    tmp_path: Path,
) -> None:
    baseline_paths = _write_track_baselines(
        tmp_path,
        ("mute_a", "mute_b", "silence_a", "silence_b", "sand_a", "sand_b"),
    )
    env = FZeroXEnv(
        backend=SyntheticBackend(),
        config=EnvConfig(
            action_repeat=1,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_a",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute_a"],
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_b",
                        course_id="mute_city",
                        baseline_state_path=baseline_paths["mute_b"],
                    ),
                    TrackSamplingEntryConfig(
                        id="silence_a",
                        course_id="silence",
                        baseline_state_path=baseline_paths["silence_a"],
                    ),
                    TrackSamplingEntryConfig(
                        id="silence_b",
                        course_id="silence",
                        baseline_state_path=baseline_paths["silence_b"],
                    ),
                    TrackSamplingEntryConfig(
                        id="sand_a",
                        course_id="sand_ocean",
                        baseline_state_path=baseline_paths["sand_a"],
                    ),
                    TrackSamplingEntryConfig(
                        id="sand_b",
                        course_id="sand_ocean",
                        baseline_state_path=baseline_paths["sand_b"],
                    ),
                ),
            ),
        ),
    )

    env.set_track_sampling_weights(
        {
            "mute_a": 0.0,
            "mute_b": 0.0,
            "silence_a": 0.0,
            "silence_b": 1.0,
            "sand_a": 0.0,
            "sand_b": 0.0,
        }
    )
    sampled_courses = [env.reset(seed=123)[1]["track_course_id"] for _ in range(4)]

    assert sampled_courses == ["silence", "silence", "silence", "silence"]


def test_curriculum_stage_can_override_track_sampling(tmp_path: Path) -> None:
    stage_zero_path = tmp_path / "mute-city.state"
    stage_one_path = tmp_path / "silence.state"
    stage_zero_path.write_bytes(b"mute")
    stage_one_path.write_bytes(b"silence")
    backend = SyntheticBackend()
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1),
        curriculum_config=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="mute",
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="mute",
                                baseline_state_path=stage_zero_path,
                                weight=1.0,
                            ),
                        ),
                    ),
                ),
                CurriculumStageConfig(
                    name="silence",
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="silence",
                                baseline_state_path=stage_one_path,
                                weight=1.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    _, first_info = env.reset(seed=123)
    env.set_curriculum_stage(1)
    _, second_info = env.reset(seed=123)

    assert backend.loaded_baselines == []
    assert backend.loaded_baseline_bytes == [
        (stage_zero_path, len(b"mute")),
        (stage_one_path, len(b"silence")),
    ]
    assert first_info["track_id"] == "mute"
    assert second_info["track_id"] == "silence"


def test_reset_randomizes_game_rng_when_enabled_and_in_race() -> None:
    backend = ScriptedStepBackend([], reset_telemetry=telemetry(race_distance=0.0))
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, randomize_game_rng_on_reset=True),
    )

    _, first_info = env.reset(seed=123)
    _, second_info = env.reset()

    assert first_info["rng_randomized"] is True
    assert second_info["rng_randomized"] is True
    assert len(backend.randomized_rng_seeds) == 2
    assert backend.randomized_rng_seeds[0] != backend.randomized_rng_seeds[1]
    assert first_info["rng_state"] != second_info["rng_state"]


def test_reset_can_randomize_game_rng_without_race_mode_requirement() -> None:
    backend = ScriptedStepBackend([], reset_telemetry=None)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            randomize_game_rng_on_reset=True,
            randomize_game_rng_requires_race_mode=False,
        ),
    )

    _, info = env.reset(seed=123)

    assert info["rng_randomized"] is True
    assert len(backend.randomized_rng_seeds) == 1


def test_reset_skips_rng_randomization_outside_race() -> None:
    backend = ScriptedStepBackend(
        [],
        reset_telemetry=make_telemetry(
            game_mode_raw=0,
            game_mode_name="title",
            in_race_mode=False,
            race_distance=0.0,
        ),
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, randomize_game_rng_on_reset=True),
    )

    _, info = env.reset(seed=123)

    assert info["rng_randomized"] is False
    assert info["rng_randomization_skip_reason"] == "not_in_race"
    assert backend.randomized_rng_seeds == []


def test_reset_applies_configured_camera_setting_with_button_loop() -> None:
    backend = CameraSyncBackend(camera_setting_raw=2)
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(action_repeat=1, camera_setting="close_behind"),
    )

    _, info = env.reset(seed=123)

    assert info["camera_setting"] == "close_behind"
    assert info["camera_setting_raw"] == 1
    assert info["camera_setting_sync"] == "changed"
    assert info["camera_setting_taps"] == 3
    assert info["frame_index"] == 6
    assert backend.last_race_control_state == RaceControlState()


def test_reset_syncs_camera_and_then_returns_to_target_intro_timer() -> None:
    backend = CameraSyncAfterIntroBackend(
        camera_setting_raw=2,
        race_intro_timer=80,
        camera_ready_intro_timer=160,
    )
    env = FZeroXEnv(
        backend=backend,
        config=EnvConfig(
            action_repeat=1,
            camera_setting="close_behind",
            race_intro_target_timer=38,
        ),
    )

    _, info = env.reset(seed=123)

    assert info["race_intro_timer_sync"] == "changed"
    assert info["race_intro_timer_target"] == 38
    assert info["camera_setting"] == "close_behind"
    assert info["camera_setting_raw"] == 1
    assert info["camera_setting_sync"] == "changed"
    assert info["race_intro_timer"] == 38


def _write_track_baselines(tmp_path: Path, track_ids: tuple[str, ...]) -> dict[str, Path]:
    baseline_paths: dict[str, Path] = {}
    for track_id in track_ids:
        baseline_path = tmp_path / f"{track_id}.state"
        baseline_path.write_bytes(track_id.encode())
        baseline_paths[track_id] = baseline_path
    return baseline_paths
