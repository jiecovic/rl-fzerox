# tests/core/training/test_training_runner.py
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
    CurriculumTriggerConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    PolicyRecurrentConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.info import MONITOR_INFO_KEYS
from rl_fzerox.core.training import runner
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.session.artifacts import (
    PolicyArtifactMetadata,
    atomic_save_artifact,
    load_policy_artifact_metadata,
    resolve_train_run_config,
    validate_training_baseline_state,
)
from rl_fzerox.core.training.session.callbacks import (
    RolloutInfoAccumulator,
    build_callbacks,
    info_sequence,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
    resolve_policy_activation_fn,
)
from tests.support.fakes import SyntheticBackend


class _CapturingLogger:
    def __init__(self) -> None:
        self.records: dict[str, object] = {}

    def record(self, key: str, value: object) -> None:
        self.records[key] = value


def test_validate_training_baseline_state_requires_existing_file(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    with pytest.raises(RuntimeError, match="Configured training baseline state"):
        validate_training_baseline_state(config)


def test_validate_training_baseline_state_accepts_existing_file(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    validate_training_baseline_state(config)


def test_validate_training_baseline_state_checks_track_sampling_entries(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    missing_baseline_path = tmp_path / "missing.state"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="missing",
                        baseline_state_path=missing_baseline_path,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    with pytest.raises(RuntimeError, match="missing.state"):
        validate_training_baseline_state(config)


def test_resolve_train_run_config_sets_run_local_runtime_root(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )

    resolved_config = resolve_train_run_config(config=config, run_paths=run_paths)

    assert resolved_config.emulator.runtime_dir == run_paths.runtime_root


def test_run_training_removes_empty_run_dir_when_config_resolution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="cleanup_probe"),
    )

    def fail_config_resolution(**_: object) -> TrainAppConfig:
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(runner, "resolve_train_run_config", fail_config_resolution)

    with pytest.raises(RuntimeError, match="materialization failed"):
        runner.run_training(config)

    assert not (tmp_path / "runs" / "cleanup_probe_0001").exists()


def test_rollout_info_accumulator_summarizes_state_and_episode_metrics() -> None:
    accumulator = RolloutInfoAccumulator()
    infos = [
        {
            "race_distance": 10.0,
            "speed_kph": 100.0,
            "position": 5,
            "lap": 1,
            "race_laps_completed": 0,
            "damage_taken_frames": 0,
            "frames_run": 2,
            "airborne_frames": 1,
            "collision_recoil_entered": False,
            "boost_used": True,
            "lean_used": False,
            "episode": {
                "position": 2,
                "race_laps_completed": 3,
                "race_time_ms": 123_400,
                "episode_step": 7_404,
                "termination_reason": "finished",
                "truncation_reason": None,
                "track_course_id": "mute_city",
            },
        },
        {
            "race_distance": 14.0,
            "speed_kph": 120.0,
            "position": 7,
            "lap": 1,
            "race_laps_completed": 0,
            "damage_taken_frames": 2,
            "frames_run": 3,
            "airborne_frames": 3,
            "collision_recoil_entered": True,
            "boost_used": False,
            "lean_used": True,
            "episode": {
                "position": 8,
                "race_laps_completed": 1,
                "race_time_ms": 40_000,
                "episode_step": 2_400,
                "termination_reason": None,
                "truncation_reason": "progress_stalled",
            },
        },
    ]

    accumulator.add_infos(infos)

    assert accumulator.state_metrics["race_distance"].mean() == 12.0
    assert accumulator.state_metrics["speed_kph"].mean() == 110.0
    assert accumulator.state_metrics["race_laps_completed"].mean() == 0.0
    assert accumulator.step_rates["damage_taken_frames"].rate() == 0.5
    assert accumulator.step_rates["collision_recoil_entered"].rate() == 0.5
    assert accumulator.step_rates["boost_used"].rate() == 0.5
    assert accumulator.step_rates["lean_used"].rate() == 0.5
    assert accumulator.frame_ratios["state/airborne_frame_ratio"].ratio() == 0.8
    assert accumulator.episode_metrics["position"].mean() == 5.0
    assert accumulator.episode_metrics["race_laps_completed"].mean() == 2.0
    assert accumulator.finished_episode_metrics["race_time_ms"].mean() == 123.4
    assert accumulator.finished_episode_metrics["episode_step"].mean() == 7404.0
    assert accumulator.finished_episode_metrics["position"].mean() == 2.0
    assert accumulator.course_finish_times_s["mute_city"].mean() == 123.4
    assert accumulator.episode_count == 2
    assert accumulator.termination_counts["finished"] == 1
    assert accumulator.truncation_counts["progress_stalled"] == 1

    logger = _CapturingLogger()
    accumulator.record_to(logger)

    assert logger.records["state/damage_taken_step_rate"] == 0.5
    assert logger.records["state/collision_recoil_entry_rate"] == 0.5
    assert logger.records["state/airborne_frame_ratio"] == 0.8
    assert logger.records["action/boost_used_step_rate"] == 0.5
    assert logger.records["action/lean_used_step_rate"] == 0.5
    assert logger.records["episode/finish_time_s_mean"] == 123.4
    assert logger.records["episode/by_course/mute_city/finish_time_s_mean"] == 123.4
    assert logger.records["episode/finish_steps_mean"] == 7404.0
    assert logger.records["episode/finish_position_mean"] == 2.0


def test_info_sequence_accepts_tuple_infos() -> None:
    infos = ({"race_distance": 10.0}, {"race_distance": 12.0})

    assert info_sequence(infos) == infos
    assert info_sequence([{"race_distance": 10.0}]) == [{"race_distance": 10.0}]
    assert info_sequence(None) is None


def test_callbacks_save_latest_artifacts_at_training_start(tmp_path: Path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    callbacks = build_callbacks(
        train_config=TrainConfig(save_freq=1_000, num_envs=1),
        curriculum_config=CurriculumConfig(),
        run_paths=run_paths,
    )
    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        assert model.num_timesteps == 0

        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})
    finally:
        env.close()

    assert run_paths.latest_model_path.is_file()
    assert run_paths.latest_policy_path.is_file()
    assert load_policy_artifact_metadata(run_paths.latest_policy_path) == PolicyArtifactMetadata(
        curriculum_stage_index=None,
        curriculum_stage_name=None,
    )


def test_resolve_policy_activation_fn_supports_known_names() -> None:
    from torch import nn

    assert resolve_policy_activation_fn("tanh") is nn.Tanh
    assert resolve_policy_activation_fn("relu") is nn.ReLU


def test_resolve_policy_activation_fn_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported policy activation"):
        resolve_policy_activation_fn("gelu")


def test_atomic_save_artifact_replaces_target_without_leaving_tmp(tmp_path: Path) -> None:
    target_path = tmp_path / "latest_policy.zip"

    def _fake_save(path: str) -> None:
        Path(path).write_bytes(b"new-policy")

    atomic_save_artifact(_fake_save, target_path)

    assert target_path.read_bytes() == b"new-policy"
    assert list(tmp_path.glob("*.tmp.zip")) == []


def test_train_config_rejects_plain_ppo_algorithm(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(ValidationError, match="algorithm"):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            policy=PolicyConfig(),
            curriculum=CurriculumConfig(),
            train=TrainConfig.model_validate({"algorithm": "ppo"}),
        )


def test_train_app_config_rejects_recurrent_policy_without_recurrent_algorithm(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match="policy.recurrent.enabled=true requires a recurrent train.algorithm",
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(
                recurrent=PolicyRecurrentConfig(enabled=True),
            ),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_ppo"),
        )


def test_train_app_config_rejects_recurrent_algorithm_without_recurrent_policy(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(
        ValidationError,
        match="train.algorithm=maskable_recurrent_ppo requires policy.recurrent.enabled=true",
    ):
        TrainAppConfig(
            emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
            env=EnvConfig(),
            policy=PolicyConfig(),
            curriculum=CurriculumConfig(),
            train=TrainConfig(algorithm="maskable_recurrent_ppo"),
        )


def test_curriculum_controller_promotes_after_smoothed_finish_threshold() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            smoothing_episodes=1,
            min_stage_episodes=1,
            stages=(
                CurriculumStageConfig(
                    name="basic_drive",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=3.0),
                    action_mask=ActionMaskConfig(lean=(0,)),
                ),
                CurriculumStageConfig(name="lean_enabled"),
            ),
        )
    )

    promoted_stage = controller.record_episodes([{"race_laps_completed": 3}])

    assert promoted_stage == 1
    assert controller.stage_index == 1
    assert controller.stage_name == "lean_enabled"


def test_curriculum_controller_exposes_active_train_overrides() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            smoothing_episodes=1,
            min_stage_episodes=1,
            stages=(
                CurriculumStageConfig(
                    name="explore",
                    until=CurriculumTriggerConfig(race_laps_completed_mean_gte=1.0),
                    train=CurriculumTrainOverridesConfig(
                        learning_rate=2.0e-4,
                        n_epochs=5,
                        batch_size=512,
                        clip_range=0.2,
                        ent_coef=0.01,
                    ),
                ),
                CurriculumStageConfig(
                    name="finetune",
                    train=CurriculumTrainOverridesConfig(
                        learning_rate=3.0e-5,
                        n_epochs=2,
                        batch_size=1024,
                        clip_range=0.1,
                        ent_coef=0.0,
                    ),
                ),
            ),
        )
    )

    assert controller.stage_train_overrides is not None
    assert controller.stage_train_overrides.learning_rate == 2.0e-4
    assert controller.stage_train_overrides.n_epochs == 5
    assert controller.stage_train_overrides.batch_size == 512
    assert controller.stage_train_overrides.clip_range == 0.2
    assert controller.stage_train_overrides.ent_coef == 0.01

    promoted_stage = controller.record_episodes([{"race_laps_completed": 1}])

    assert promoted_stage == 1
    assert controller.stage_train_overrides is not None
    assert controller.stage_train_overrides.learning_rate == 3.0e-5
    assert controller.stage_train_overrides.n_epochs == 2
    assert controller.stage_train_overrides.batch_size == 1024
    assert controller.stage_train_overrides.clip_range == 0.1
    assert controller.stage_train_overrides.ent_coef == 0.0


def test_curriculum_callback_applies_stage_train_overrides(tmp_path: Path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    curriculum = CurriculumConfig(
        enabled=True,
        smoothing_episodes=1,
        min_stage_episodes=1,
        stages=(
            CurriculumStageConfig(
                name="explore",
                until=CurriculumTriggerConfig(race_laps_completed_mean_gte=1.0),
                train=CurriculumTrainOverridesConfig(
                    learning_rate=2.0e-4,
                    n_epochs=5,
                    batch_size=4,
                    clip_range=0.2,
                    ent_coef=0.01,
                ),
            ),
            CurriculumStageConfig(
                name="finetune",
                train=CurriculumTrainOverridesConfig(
                    learning_rate=3.0e-5,
                    n_epochs=2,
                    batch_size=4,
                    clip_range=0.1,
                    ent_coef=0.0,
                ),
            ),
        ),
    )
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    callbacks = build_callbacks(
        train_config=TrainConfig(save_freq=1_000, num_envs=1),
        curriculum_config=curriculum,
        run_paths=run_paths,
    )
    env = DummyVecEnv(
        [
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
                curriculum_config=curriculum,
            )
        ]
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
                ent_coef=0.0,
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})

        assert model.lr_schedule(1.0) == pytest.approx(2.0e-4)
        assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)
        assert model.n_epochs == 5
        assert model.batch_size == 4
        clip_range = model.clip_range
        assert callable(clip_range)
        assert clip_range(1.0) == pytest.approx(0.2)
        assert model.ent_coef == pytest.approx(0.01)

        callbacks.update_locals(
            {
                "infos": [
                    {
                        "episode": {
                            "race_laps_completed": 1,
                        }
                    }
                ]
            }
        )
        callbacks.on_step()

        assert model.lr_schedule(1.0) == pytest.approx(3.0e-5)
        assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-5)
        assert model.n_epochs == 2
        assert model.batch_size == 4
        clip_range = model.clip_range
        assert callable(clip_range)
        assert clip_range(1.0) == pytest.approx(0.1)
        assert model.ent_coef == pytest.approx(0.0)
    finally:
        env.close()


def test_monitor_info_keys_include_position_context() -> None:
    assert "position" in MONITOR_INFO_KEYS
    assert "total_racers" in MONITOR_INFO_KEYS
    assert "course_index" in MONITOR_INFO_KEYS


def test_monitor_info_keys_include_finished_timing_and_collision_metrics() -> None:
    assert "race_time_ms" in MONITOR_INFO_KEYS
    assert "damage_taken_frames" in MONITOR_INFO_KEYS
    assert "collision_recoil_entered" in MONITOR_INFO_KEYS


def test_monitor_info_keys_exclude_step_only_action_rates() -> None:
    assert "boost_used" not in MONITOR_INFO_KEYS
    assert "lean_used" not in MONITOR_INFO_KEYS


def test_monitor_info_keys_include_track_context_for_course_metrics() -> None:
    assert "track_id" in MONITOR_INFO_KEYS
    assert "track_course_id" in MONITOR_INFO_KEYS
    assert "track_course_name" in MONITOR_INFO_KEYS
