# tests/core/envs/test_reset.py
import pickle
from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    CurriculumStageConfig,
    EnvConfig,
    ObservationConfig,
    TrackRecordEntryConfig,
    TrackRecordsConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.engine.reset import TrackBaselineCache
from tests.core.envs.helpers import (
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

    assert obs.shape == (116, 164, 12)
    assert obs.dtype == np.uint8
    assert info["backend"] == "synthetic"
    assert info["seed"] == 123
    assert info["observation_shape"] == (116, 164, 12)
    assert info["observation_frame_shape"] == (116, 164, 3)
    assert info["observation_stack"] == 4
    assert np.array_equal(obs[:, :, 0:3], obs[:, :, 3:6])
    assert np.array_equal(obs[:, :, 3:6], obs[:, :, 6:9])
    assert np.array_equal(obs[:, :, 6:9], obs[:, :, 9:12])
    assert isinstance(env.action_space, MultiDiscrete)
    assert env.action_space.nvec.tolist() == [7, 3, 2, 3]


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
            observation=ObservationConfig(mode="image_state", frame_stack=4),
        ),
    )

    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert set(obs) == {"image", "state"}
    assert obs["image"].shape == (116, 164, 12)
    assert obs["image"].dtype == np.uint8
    assert obs["state"].shape == (11,)
    assert obs["state"].dtype == np.float32
    assert obs["state"].tolist() == pytest.approx(
        [0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    )
    assert info["observation_mode"] == "image_state"
    assert info["observation_shape"] == (116, 164, 12)
    assert info["observation_state_shape"] == (11,)
    assert info["observation_state_features"] == (
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "can_boost",
        "boost_active",
        "left_lean_held",
        "right_lean_held",
        "left_press_age_norm",
        "right_press_age_norm",
        "recent_boost_pressure",
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
    env._engine._episode_done = True
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
    assert backend.last_controller_state == ControllerState()


def _write_track_baselines(tmp_path: Path, track_ids: tuple[str, ...]) -> dict[str, Path]:
    baseline_paths: dict[str, Path] = {}
    for track_id in track_ids:
        baseline_path = tmp_path / f"{track_id}.state"
        baseline_path.write_bytes(track_id.encode())
        baseline_paths[track_id] = baseline_path
    return baseline_paths
