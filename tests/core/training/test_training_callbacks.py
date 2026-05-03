# tests/core/training/test_training_callbacks.py
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_fzerox.core.config.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    EnvConfig,
    PolicyConfig,
    TrainConfig,
)
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.session.artifacts import list_recent_checkpoint_dirs
from rl_fzerox.core.training.session.callbacks import build_callbacks
from rl_fzerox.core.training.session.callbacks.checkpoints import resolve_checkpoint_policy
from rl_fzerox.core.training.session.model import build_ppo_model
from tests.support.fakes import SyntheticBackend


def test_resolve_checkpoint_policy_prefers_rollout_interval_for_ppo() -> None:
    policy = resolve_checkpoint_policy(
        TrainConfig(
            algorithm="maskable_hybrid_recurrent_ppo",
            checkpoint_every_rollouts=4,
            num_envs=10,
            save_freq=1_000,
            save_recent_checkpoints=True,
            recent_checkpoint_limit=3,
        )
    )

    assert policy.rollout_interval == 4
    assert policy.step_interval is None
    assert policy.save_recent is True
    assert policy.recent_limit == 3


def test_resolve_checkpoint_policy_falls_back_to_step_interval_for_sac() -> None:
    policy = resolve_checkpoint_policy(
        TrainConfig(
            algorithm="hybrid_action_sac",
            checkpoint_every_rollouts=4,
            num_envs=10,
            save_freq=1_000,
        )
    )

    assert policy.rollout_interval is None
    assert policy.step_interval == 100


def test_rollout_checkpoint_callback_saves_and_trims_recent_snapshots(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    callbacks = build_callbacks(
        train_config=TrainConfig(
            algorithm="maskable_ppo",
            num_envs=1,
            n_steps=4,
            batch_size=4,
            checkpoint_every_rollouts=1,
            save_latest_checkpoint=True,
            save_best_checkpoint=False,
            save_recent_checkpoints=True,
            recent_checkpoint_limit=1,
        ),
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
                num_envs=1,
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        model.set_logger(configure(folder=None, format_strings=[]))
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})
        model.num_timesteps = 4
        callbacks.on_rollout_end()
        model.num_timesteps = 8
        callbacks.on_rollout_end()
    finally:
        env.close()

    checkpoint_dirs = list_recent_checkpoint_dirs(run_paths)
    assert [path.name for path in checkpoint_dirs] == ["000000000008"]
    assert run_paths.latest_model_path.is_file()
    assert _checkpoint_files(checkpoint_dirs[-1]) == {
        "model.zip",
        "policy.metadata.json",
        "policy.zip",
    }


def _checkpoint_files(path: Path) -> set[str]:
    return {child.name for child in path.iterdir()}
