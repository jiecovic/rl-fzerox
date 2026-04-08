# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import (
    build_run_paths,
    ensure_run_dirs,
    save_train_run_config,
)
from rl_fzerox.core.training.session.artifacts import (
    _cleanup_failed_run,
    _resolve_train_run_config,
    _save_artifacts_atomically,
    _save_latest_artifacts,
    _validate_training_baseline_state,
)
from rl_fzerox.core.training.session.callbacks import _build_callbacks
from rl_fzerox.core.training.session.env import _build_training_env
from rl_fzerox.core.training.session.model import (
    _build_ppo_model,
    _build_tensorboard_logger,
    _print_training_startup,
)


def run_training(config: TrainAppConfig) -> None:
    """Run one PPO training session from the composed train config."""

    seed_process(config.seed)
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    _validate_training_baseline_state(config)

    ensure_run_dirs(run_paths)
    run_config = _resolve_train_run_config(config=config, run_paths=run_paths)
    train_env = None
    model = None

    try:
        train_env = _build_training_env(run_config, run_paths)
        model = _build_ppo_model(
            train_env=train_env,
            train_config=run_config.train,
            policy_config=run_config.policy,
            tensorboard_log=None,
        )
        save_train_run_config(config=run_config, run_dir=run_paths.run_dir)
        model.set_logger(_build_tensorboard_logger(run_paths))
        _print_training_startup(
            model=model,
            train_env=train_env,
            config=run_config,
            run_paths=run_paths,
        )
        _save_latest_artifacts(model, run_paths)
        callbacks = _build_callbacks(
            train_config=run_config.train,
            run_paths=run_paths,
        )
        try:
            model.learn(
                total_timesteps=run_config.train.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
        except Exception:
            if model.num_timesteps > 0:
                _save_latest_artifacts(model, run_paths)
            raise
        _save_artifacts_atomically(
            model=model,
            model_path=run_paths.final_model_path,
            policy_path=run_paths.final_policy_path,
        )
        _save_latest_artifacts(model, run_paths)
    except Exception:
        _cleanup_failed_run(run_paths, model)
        raise
    finally:
        if train_env is not None:
            train_env.close()
