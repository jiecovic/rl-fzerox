# tests/core/training/test_training_runner_curriculum.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    CurriculumTrainOverridesConfig,
    CurriculumTriggerConfig,
    EnvConfig,
    PerTrackLapsCompletedTriggerConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.session.callbacks import (
    build_callbacks,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController
from rl_fzerox.core.training.session.model import (
    build_ppo_model,
)
from tests.support.fakes import SyntheticBackend, vec_env_fns


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


def test_curriculum_controller_requires_per_track_lap_coverage() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            smoothing_episodes=4,
            min_stage_episodes=3,
            stages=(
                CurriculumStageConfig(
                    name="jack",
                    until=CurriculumTriggerConfig(
                        per_track_laps_completed=PerTrackLapsCompletedTriggerConfig(
                            mean_gte=0.5,
                            min_track_fraction_gte=1.0,
                            min_episodes_per_track=1,
                        )
                    ),
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(id="a", course_id="a"),
                            TrackSamplingEntryConfig(id="b", course_id="b"),
                            TrackSamplingEntryConfig(id="c", course_id="c"),
                        ),
                    ),
                ),
                CurriculumStageConfig(name="queen_seed"),
            ),
        )
    )

    assert (
        controller.record_episodes(
            [
                {"track_course_id": "a", "race_laps_completed": 1},
                {"track_course_id": "b", "race_laps_completed": 1},
                {"track_course_id": "a", "race_laps_completed": 1},
            ]
        )
        is None
    )

    promoted_stage = controller.record_episodes(
        [{"track_course_id": "c", "race_laps_completed": 1}]
    )

    assert promoted_stage == 1
    assert controller.stage_name == "queen_seed"


def test_curriculum_controller_allows_per_track_fraction_gate() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            smoothing_episodes=4,
            min_stage_episodes=3,
            stages=(
                CurriculumStageConfig(
                    name="jack",
                    until=CurriculumTriggerConfig(
                        per_track_laps_completed=PerTrackLapsCompletedTriggerConfig(
                            mean_gte=0.5,
                            min_track_fraction_gte=0.5,
                            min_episodes_per_track=1,
                        )
                    ),
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(id="a", course_id="a"),
                            TrackSamplingEntryConfig(id="b", course_id="b"),
                            TrackSamplingEntryConfig(id="c", course_id="c"),
                            TrackSamplingEntryConfig(id="d", course_id="d"),
                        ),
                    ),
                ),
                CurriculumStageConfig(name="queen_seed"),
            ),
        )
    )

    promoted_stage = controller.record_episodes(
        [
            {"track_course_id": "a", "race_laps_completed": 1},
            {"track_course_id": "b", "race_laps_completed": 1},
            {"track_course_id": "c", "race_laps_completed": 0},
        ]
    )

    assert promoted_stage == 1


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


def test_curriculum_controller_can_start_from_checkpoint_stage() -> None:
    controller = ActionMaskCurriculumController(
        CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(name="explore"),
                CurriculumStageConfig(name="finetune"),
            ),
        ),
        initial_stage_index=1,
    )

    assert controller.stage_index == 1
    assert controller.stage_name == "finetune"


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
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
                curriculum_config=curriculum,
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


def test_curriculum_callback_starts_from_resume_stage(tmp_path: Path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    curriculum = CurriculumConfig(
        enabled=True,
        stages=(
            CurriculumStageConfig(
                name="explore",
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
        initial_curriculum_stage_index=1,
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
                curriculum_config=curriculum,
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
                ent_coef=0.01,
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})

        assert env.get_attr("curriculum_stage_index") == [1]
        assert model.lr_schedule(1.0) == pytest.approx(3.0e-5)
        assert model.n_epochs == 2
        assert model.ent_coef == pytest.approx(0.0)
    finally:
        env.close()
