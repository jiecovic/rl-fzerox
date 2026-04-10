# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs, save_train_run_config
from rl_fzerox.core.training.session import (
    build_callbacks,
    build_ppo_model,
    build_tensorboard_logger,
    build_training_env,
    cleanup_failed_run,
    current_policy_artifact_metadata,
    maybe_preload_training_parameters,
    print_training_startup,
    resolve_train_run_config,
    save_artifacts_atomically,
    save_latest_artifacts,
    training_requires_action_masks,
    validate_training_algorithm_config,
    validate_training_baseline_state,
)


def run_training(config: TrainAppConfig) -> None:
    """Run one PPO training session from the composed train config."""

    seed_process(config.seed)
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    validate_training_baseline_state(config)
    validate_training_algorithm_config(config)

    ensure_run_dirs(run_paths)
    run_config = resolve_train_run_config(config=config, run_paths=run_paths)
    train_env = None
    model = None

    try:
        train_env = build_training_env(run_config, run_paths)
        model = build_ppo_model(
            train_env=train_env,
            train_config=run_config.train,
            policy_config=run_config.policy,
            tensorboard_log=None,
            masking_required=training_requires_action_masks(run_config),
        )
        maybe_preload_training_parameters(
            model=model,
            train_config=run_config.train,
        )
        save_train_run_config(config=run_config, run_dir=run_paths.run_dir)
        model.set_logger(build_tensorboard_logger(run_paths))
        print_training_startup(
            model=model,
            train_env=train_env,
            config=run_config,
            run_paths=run_paths,
        )
        save_latest_artifacts(
            model,
            run_paths,
            policy_metadata=current_policy_artifact_metadata(train_env),
        )
        callbacks = build_callbacks(
            train_config=run_config.train,
            curriculum_config=run_config.curriculum,
            run_paths=run_paths,
        )
        masking_required = training_requires_action_masks(run_config)
        try:
            _learn_model(
                model=model,
                total_timesteps=run_config.train.total_timesteps,
                callback=callbacks,
                use_masking=masking_required,
                progress_bar=True,
            )
        except Exception:
            if model.num_timesteps > 0:
                save_latest_artifacts(
                    model,
                    run_paths,
                    policy_metadata=current_policy_artifact_metadata(train_env),
                )
            raise
        save_artifacts_atomically(
            model=model,
            model_path=run_paths.final_model_path,
            policy_path=run_paths.final_policy_path,
            policy_metadata=current_policy_artifact_metadata(train_env),
        )
        save_latest_artifacts(
            model,
            run_paths,
            policy_metadata=current_policy_artifact_metadata(train_env),
        )
    except Exception:
        cleanup_failed_run(run_paths, model)
        raise
    finally:
        if train_env is not None:
            train_env.close()


def _learn_model(
    *,
    model,
    total_timesteps: int,
    callback,
    use_masking: bool,
    progress_bar: bool,
) -> None:
    """Train one mask-aware PPO-family model with the shared learn kwargs."""

    from sb3_contrib import MaskablePPO

    if isinstance(model, MaskablePPO):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
        )
        return

    try:
        from sb3x import MaskableRecurrentPPO
    except ImportError:
        MaskableRecurrentPPO = None

    if MaskableRecurrentPPO is not None and isinstance(model, MaskableRecurrentPPO):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
        )
        return

    raise TypeError(f"Unsupported training model type: {type(model).__name__}")
