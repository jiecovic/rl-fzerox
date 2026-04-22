# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import (
    ensure_run_dirs,
    reserve_run_paths,
    resolve_policy_artifact_path,
    save_train_run_config,
)
from rl_fzerox.core.training.session import (
    build_callbacks,
    build_tensorboard_logger,
    build_training_env,
    build_training_model,
    cleanup_failed_run,
    current_policy_artifact_metadata,
    load_policy_artifact_metadata,
    maybe_resume_training_model,
    print_training_startup,
    resolve_train_run_config,
    save_artifacts_atomically,
    save_latest_artifacts,
    training_requires_action_masks,
    validate_training_algorithm_config,
    validate_training_baseline_state,
)


def run_training(config: TrainAppConfig) -> None:
    """Run one training session from the composed train config."""

    seed_process(config.seed)
    run_paths = reserve_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    validate_training_algorithm_config(config)
    train_env = None
    model = None

    try:
        ensure_run_dirs(run_paths)
        run_config = resolve_train_run_config(config=config, run_paths=run_paths)
        validate_training_baseline_state(run_config)
        train_env = build_training_env(run_config, run_paths)
        model = build_training_model(
            train_env=train_env,
            train_config=run_config.train,
            policy_config=run_config.policy,
            tensorboard_log=None,
        )
        model = maybe_resume_training_model(
            model=model,
            train_env=train_env,
            train_config=run_config.train,
        )
        initial_curriculum_stage_index = _resume_curriculum_stage_index(run_config)
        if initial_curriculum_stage_index is not None:
            train_env.env_method(
                "sync_checkpoint_curriculum_stage",
                initial_curriculum_stage_index,
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
            initial_curriculum_stage_index=initial_curriculum_stage_index,
        )
        masking_required = training_requires_action_masks(run_config)
        try:
            _learn_model(
                model=model,
                total_timesteps=run_config.train.total_timesteps,
                callback=callbacks,
                use_masking=masking_required,
                progress_bar=True,
                reset_num_timesteps=run_config.train.resume_mode != "full_model",
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


def _resume_curriculum_stage_index(config: TrainAppConfig) -> int | None:
    if config.train.resume_run_dir is None or config.train.resume_mode != "full_model":
        return None
    policy_path = resolve_policy_artifact_path(
        config.train.resume_run_dir,
        artifact=config.train.resume_artifact,
    )
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None:
        return None
    return metadata.curriculum_stage_index


def _learn_model(
    *,
    model,
    total_timesteps: int,
    callback,
    use_masking: bool,
    progress_bar: bool,
    reset_num_timesteps: bool,
) -> None:
    """Train one configured SB3 model with the shared learn kwargs."""

    from sb3_contrib import MaskablePPO
    from stable_baselines3 import SAC

    if isinstance(model, SAC):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        return

    if isinstance(model, MaskablePPO):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
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
            reset_num_timesteps=reset_num_timesteps,
        )
        return

    try:
        from sb3x import MaskableHybridRecurrentPPO
    except ImportError:
        MaskableHybridRecurrentPPO = None

    if MaskableHybridRecurrentPPO is not None and isinstance(model, MaskableHybridRecurrentPPO):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        return

    try:
        from sb3x import MaskableHybridActionPPO
    except ImportError:
        MaskableHybridActionPPO = None

    if MaskableHybridActionPPO is not None and isinstance(model, MaskableHybridActionPPO):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        return

    raise TypeError(f"Unsupported training model type: {type(model).__name__}")
