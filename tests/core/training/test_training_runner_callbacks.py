# tests/core/training/test_training_runner_callbacks.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training import runner
from rl_fzerox.core.training.runs import RUN_LAYOUT, build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.session.artifacts import (
    PolicyArtifactMetadata,
    load_policy_artifact_metadata,
    policy_artifact_metadata_path,
)
from rl_fzerox.core.training.session.callbacks import (
    RolloutInfoAccumulator,
    build_callbacks,
    info_sequence,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingMaterializedArtifact,
    TrackSamplingRuntimePersistence,
)
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
)
from tests.core.training.training_runner_support import (
    _CapturingLogger,
    _full_model_resume_config,
    _RunTrainingEnv,
    _RunTrainingModel,
    _stub_run_training_dependencies,
    _TrainingStepModel,
)
from tests.support.fakes import SyntheticBackend, vec_env_fns


def test_rollout_info_accumulator_summarizes_state_and_episode_metrics() -> None:
    accumulator = RolloutInfoAccumulator()
    infos = [
        {
            "race_distance": 10.0,
            "speed_kph": 100.0,
            "position": 5,
            "ko_star_count": 3,
            "ko_star_reward_event": True,
            "lap": 1,
            "race_laps_completed": 0,
            "step_reward_raw": 100.0,
            "step_reward_clipped": True,
            "step_reward_clip_abs_excess": 25.0,
            "step_reward_clip_positive": True,
            "step_reward_clip_negative": False,
            "damage_taken_frames": 0,
            "frames_run": 2,
            "airborne_frames": 1,
            "boost_pad_entered": True,
            "gas_level": 1.0,
            "steer_level": -0.25,
            "lean_level": -1.0,
            "lean_request_level": -1.0,
            "lean_episode_masked": True,
            "air_brake_episode_masked": True,
            "spin_episode_masked": False,
            "gas_used": True,
            "air_brake_requested": True,
            "air_brake_used": False,
            "boost_used": True,
            "lean_used": False,
            "spin_requested": True,
            "spin_started": True,
            "spin_macro_active_frames": 1,
            "episode": {
                "position": 2,
                "race_laps_completed": 3,
                "boost_pad_entries": 6,
                "boost_pad_entries_per_lap": 2.0,
                "race_time_ms": 123_400,
                "episode_step": 7_404,
                "episode_airborne_frames": 2,
                "lean_episode_masked": True,
                "air_brake_episode_masked": True,
                "spin_episode_masked": False,
                "termination_reason": "finished",
                "truncation_reason": None,
                "track_course_id": "mute_city",
            },
        },
        {
            "race_distance": 14.0,
            "speed_kph": 120.0,
            "position": 7,
            "ko_star_count": 3,
            "lap": 1,
            "race_laps_completed": 0,
            "step_reward_raw": -50.0,
            "step_reward_clipped": False,
            "step_reward_clip_abs_excess": 0.0,
            "step_reward_clip_positive": False,
            "step_reward_clip_negative": False,
            "damage_taken_frames": 2,
            "frames_run": 3,
            "airborne_frames": 3,
            "boost_pad_entered": False,
            "gas_level": 0.0,
            "steer_level": 0.75,
            "lean_level": 1.0,
            "lean_request_level": 0.0,
            "lean_episode_masked": False,
            "air_brake_episode_masked": False,
            "spin_episode_masked": True,
            "gas_used": False,
            "air_brake_requested": False,
            "air_brake_used": True,
            "boost_used": False,
            "lean_used": True,
            "spin_requested": False,
            "spin_started": False,
            "spin_macro_active_frames": 0,
            "episode": {
                "position": 8,
                "race_laps_completed": 1,
                "boost_pad_entries": 1,
                "boost_pad_entries_per_lap": 1.0,
                "race_time_ms": 40_000,
                "episode_step": 2_400,
                "episode_airborne_frames": 3,
                "lean_episode_masked": False,
                "air_brake_episode_masked": False,
                "spin_episode_masked": True,
                "termination_reason": None,
                "truncation_reason": "progress_stalled",
            },
        },
    ]

    accumulator.add_infos(infos)

    assert accumulator.state_metrics["race_distance"].mean() == 12.0
    assert accumulator.state_metrics["speed_kph"].mean() == 110.0
    assert accumulator.state_metrics["ko_star_count"].mean() == 3.0
    assert accumulator.state_metrics["race_laps_completed"].mean() == 0.0
    assert accumulator.state_metrics["step_reward_raw"].mean() == 25.0
    assert accumulator.state_metrics["step_reward_clip_abs_excess"].mean() == 12.5
    assert accumulator.state_metrics["gas_level"].mean() == 0.5
    assert accumulator.state_metrics["steer_level"].mean() == 0.25
    assert accumulator.state_metrics["lean_level"].mean() == 0.0
    assert accumulator.state_metrics["lean_request_level"].mean() == -0.5
    assert accumulator.step_rates["damage_taken_frames"].rate() == 0.5
    assert accumulator.step_rates["boost_pad_entered"].rate() == 0.5
    assert accumulator.step_rates["gas_used"].rate() == 0.5
    assert accumulator.step_rates["air_brake_requested"].rate() == 0.5
    assert accumulator.step_rates["air_brake_used"].rate() == 0.5
    assert accumulator.step_rates["boost_used"].rate() == 0.5
    assert accumulator.step_rates["lean_used"].rate() == 0.5
    assert accumulator.step_rates["lean_episode_masked"].rate() == 0.5
    assert accumulator.step_rates["air_brake_episode_masked"].rate() == 0.5
    assert accumulator.step_rates["spin_episode_masked"].rate() == 0.5
    assert accumulator.step_rates["spin_requested"].rate() == 0.5
    assert accumulator.step_rates["spin_started"].rate() == 0.5
    assert accumulator.step_rates["ko_star_reward_event"].rate() == 0.5
    assert accumulator.step_rates["step_reward_clipped"].rate() == 0.5
    assert accumulator.step_rates["step_reward_clip_positive"].rate() == 0.5
    assert accumulator.step_rates["step_reward_clip_negative"].rate() == 0.0
    assert accumulator.frame_ratios["state/airborne_frame_ratio"].ratio() == 0.8
    assert accumulator.frame_ratios["action/spin_macro_frame_ratio"].ratio() == 0.2
    assert accumulator.episode_metrics["position"].mean() == 5.0
    assert accumulator.episode_metrics["race_laps_completed"].mean() == 2.0
    assert accumulator.episode_metrics["boost_pad_entries"].mean() == 3.5
    assert accumulator.episode_metrics["boost_pad_entries_per_lap"].mean() == 1.5
    assert accumulator.finished_episode_metrics["race_time_ms"].mean() == 123.4
    assert accumulator.finished_episode_metrics["episode_step"].mean() == 7404.0
    assert accumulator.finished_episode_metrics["position"].mean() == 2.0
    assert accumulator.course_finish_times_s["mute_city"].mean() == 123.4
    assert accumulator.episode_count == 2
    assert accumulator.airborne_episode_count == 2
    assert accumulator.airborne_finished_count == 1
    assert accumulator.airborne_failed_count == 1
    assert accumulator.lean_masked_episode_count == 1
    assert accumulator.air_brake_masked_episode_count == 1
    assert accumulator.spin_masked_episode_count == 1
    assert accumulator.termination_counts["finished"] == 1
    assert accumulator.truncation_counts["progress_stalled"] == 1

    logger = _CapturingLogger()
    accumulator.record_to(logger)

    assert logger.records["state/damage_taken_step_rate"] == 0.5
    assert logger.records["state/boost_pad_entry_step_rate"] == 0.5
    assert logger.records["state/airborne_frame_ratio"] == 0.8
    assert logger.records["state/ko_star_count_mean"] == 3.0
    assert logger.records["action/gas_level_mean"] == 0.5
    assert logger.records["action/steer_level_mean"] == 0.25
    assert logger.records["action/lean_level_mean"] == 0.0
    assert logger.records["action/lean_request_level_mean"] == -0.5
    assert logger.records["action/gas_used_step_rate"] == 0.5
    assert logger.records["action/air_brake_requested_step_rate"] == 0.5
    assert logger.records["action/air_brake_used_step_rate"] == 0.5
    assert logger.records["action/boost_used_step_rate"] == 0.5
    assert logger.records["action/lean_used_step_rate"] == 0.5
    assert logger.records["action/lean_episode_masked_step_rate"] == 0.5
    assert logger.records["action/air_brake_episode_masked_step_rate"] == 0.5
    assert logger.records["action/spin_episode_masked_step_rate"] == 0.5
    assert logger.records["action/spin_requested_step_rate"] == 0.5
    assert logger.records["action/spin_started_step_rate"] == 0.5
    assert logger.records["action/spin_macro_frame_ratio"] == 0.2
    assert logger.records["reward/step_raw_mean"] == 25.0
    assert logger.records["reward/ko_star_event_step_rate"] == 0.5
    assert logger.records["reward_clip/abs_excess_mean"] == 12.5
    assert logger.records["reward_clip/any_step_rate"] == 0.5
    assert logger.records["reward_clip/positive_step_rate"] == 0.5
    assert logger.records["reward_clip/negative_step_rate"] == 0.0
    assert logger.records["episode/boost_pad_entries_mean"] == 3.5
    assert logger.records["episode/boost_pad_entries_per_lap_mean"] == 1.5
    assert logger.records["episode/finish_time_s_mean"] == 123.4
    assert logger.records["episode/by_course/mute_city/finish_time_s_mean"] == 123.4
    assert logger.records["episode/finish_steps_mean"] == 7404.0
    assert logger.records["episode/finish_position_mean"] == 2.0
    assert logger.records["episode/airborne_episode_rate"] == 1.0
    assert logger.records["episode/airborne_finish_rate"] == 0.5
    assert logger.records["episode/airborne_failure_rate"] == 0.5
    assert logger.records["episode/lean_episode_masked_rate"] == 0.5
    assert logger.records["episode/air_brake_episode_masked_rate"] == 0.5
    assert logger.records["episode/spin_episode_masked_rate"] == 0.5


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
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
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
        num_timesteps=0,
    )


def test_resume_curriculum_stage_index_reads_full_model_artifact_metadata(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    latest_policy_path = run_dir / RUN_LAYOUT.policy_artifacts.latest
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    latest_policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(latest_policy_path).write_text(
        json.dumps(
            {
                "curriculum_stage_index": 2,
                "curriculum_stage_name": "finetune",
            }
        ),
        encoding="utf-8",
    )

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(
            resume_run_dir=run_dir,
            resume_artifact="latest",
            resume_mode="full_model",
        ),
    )

    assert runner._resume_curriculum_stage_index(config) == 2


def test_weights_only_resume_does_not_restore_curriculum_stage(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)

    config = TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        policy=PolicyConfig(),
        curriculum=CurriculumConfig(),
        train=TrainConfig(
            resume_run_dir=run_dir,
            resume_artifact="latest",
            resume_mode="weights_only",
        ),
    )

    assert runner._resume_curriculum_stage_index(config) is None


def test_learn_total_timesteps_uses_full_target_when_resetting_counter() -> None:
    model = _TrainingStepModel(num_timesteps=900)

    total_timesteps = runner._learn_total_timesteps(
        model=model,
        configured_total_timesteps=1_000,
        reset_num_timesteps=True,
    )

    assert total_timesteps == 1_000


def test_learn_total_timesteps_uses_remaining_steps_for_full_model_resume() -> None:
    model = _TrainingStepModel(num_timesteps=900)

    total_timesteps = runner._learn_total_timesteps(
        model=model,
        configured_total_timesteps=1_000,
        reset_num_timesteps=False,
    )

    assert total_timesteps == 100


def test_learn_total_timesteps_clamps_completed_full_model_resume() -> None:
    model = _TrainingStepModel(num_timesteps=1_020)

    total_timesteps = runner._learn_total_timesteps(
        model=model,
        configured_total_timesteps=1_000,
        reset_num_timesteps=False,
    )

    assert total_timesteps == 0


def test_run_training_full_model_resume_learns_remaining_timesteps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_model_resume_config(tmp_path, total_timesteps=1_000)
    env = _RunTrainingEnv()
    model = _RunTrainingModel(num_timesteps=900)
    captured_learn_kwargs: dict[str, object] = {}

    _stub_run_training_dependencies(monkeypatch, config=config, env=env, model=model)

    def capture_learn_model(**kwargs: object) -> None:
        captured_learn_kwargs.update(kwargs)

    monkeypatch.setattr(runner, "_learn_model", capture_learn_model)

    runner.run_training(config)

    assert captured_learn_kwargs["model"] is model
    assert captured_learn_kwargs["total_timesteps"] == 100
    assert captured_learn_kwargs["reset_num_timesteps"] is False
    assert env.closed


def test_run_training_full_model_resume_skips_completed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_model_resume_config(tmp_path, total_timesteps=1_000)
    env = _RunTrainingEnv()
    model = _RunTrainingModel(num_timesteps=1_000)
    learn_called = False

    _stub_run_training_dependencies(monkeypatch, config=config, env=env, model=model)

    def capture_learn_model(**_: object) -> None:
        nonlocal learn_called
        learn_called = True

    monkeypatch.setattr(runner, "_learn_model", capture_learn_model)

    runner.run_training(config)

    assert not learn_called
    assert env.closed


def test_run_training_publishes_initial_track_sampling_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_model_resume_config(tmp_path, total_timesteps=1_000)
    run_dir = config.train.continue_run_dir
    assert run_dir is not None
    baseline_path = run_dir / "baselines" / "x_cup.state"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_bytes(b"state")
    config = config.model_copy(
        update={
            "env": config.env.model_copy(
                update={
                    "track_sampling": TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="x_cup_abcd1234",
                                runtime_course_key="x_cup_slot_1",
                                course_id="x_cup_abcd1234",
                                course_name="X Cup abcd1234",
                                baseline_state_path=baseline_path,
                                course_index=X_CUP_COURSE.course_index,
                                mode=X_CUP_COURSE.race_mode,
                                gp_difficulty="novice",
                                vehicle="blue_falcon",
                                generated_course_kind=X_CUP_COURSE.generated_kind,
                                generated_course_seed=1234,
                                generated_course_hash="abcd1234",
                                generated_course_slot=1,
                                generated_course_generation=2,
                            ),
                        ),
                    )
                }
            )
        }
    )
    env = _RunTrainingEnv()
    model = _RunTrainingModel(num_timesteps=900)
    captured_artifacts: list[tuple[TrackSamplingMaterializedArtifact, ...]] = []
    captured_slots: list[tuple[GeneratedXCupSlot, ...]] = []
    persistence = TrackSamplingRuntimePersistence(
        load=lambda: None,
        save=lambda _: None,
        replace_materialized_artifacts=captured_artifacts.append,
        replace_generated_x_cup_slots=captured_slots.append,
    )
    _stub_run_training_dependencies(monkeypatch, config=config, env=env, model=model)
    monkeypatch.setattr(runner, "_learn_model", lambda **_: None)

    runner.run_training(config, track_sampling_runtime_persistence=persistence)

    assert captured_artifacts
    assert captured_artifacts[0][0].course_key == "x_cup_slot_1"
    assert captured_artifacts[0][0].baseline_state_path == baseline_path.resolve()
    assert captured_slots == [
        (
            GeneratedXCupSlot(
                course_key="x_cup_slot_1",
                slot=1,
                generation=2,
                course_id="x_cup_abcd1234",
                course_name="X Cup abcd1234",
                course_hash="abcd1234",
                course_seed=1234,
                segment_count=None,
                course_length=None,
            ),
        ),
    ]
