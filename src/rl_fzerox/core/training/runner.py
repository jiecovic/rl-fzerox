# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import attrgetter

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import (
    continue_run_paths,
    ensure_run_dirs,
    explicit_run_paths,
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


def run_training(
    config: TrainAppConfig,
    *,
    extra_callbacks: Sequence[object] = (),
    startup_reporter: Callable[[str, str], None] | None = None,
) -> None:
    """Run one training session from the composed train config."""

    seed_process(config.seed)
    run_paths = (
        continue_run_paths(config.train.continue_run_dir)
        if config.train.continue_run_dir is not None
        else (
            explicit_run_paths(config.train.explicit_run_dir)
            if config.train.explicit_run_dir is not None
            else reserve_run_paths(
                output_root=config.train.output_root,
                run_name=config.train.run_name,
            )
        )
    )
    _report_startup(
        startup_reporter,
        "startup_prepare",
        f"Using run directory {run_paths.run_dir}",
    )
    validate_training_algorithm_config(config)
    train_env = None
    model = None

    try:
        ensure_run_dirs(run_paths)
        _report_startup(
            startup_reporter,
            "startup_prepare",
            "Resolving run-local config and baseline state",
        )
        run_config = resolve_train_run_config(
            config=config,
            run_paths=run_paths,
            startup_reporter=startup_reporter,
        )
        validate_training_baseline_state(run_config)
        _report_startup(
            startup_reporter,
            "startup_prepare",
            "Building training environments",
        )
        train_env = build_training_env(run_config, run_paths)
        _report_startup(
            startup_reporter,
            "startup_prepare",
            "Building policy and optimizer",
        )
        model = build_training_model(
            train_env=train_env,
            train_config=run_config.train,
            policy_config=run_config.policy,
            tensorboard_log=None,
        )
        if run_config.train.resume_run_dir is not None:
            _report_startup(
                startup_reporter,
                "startup_resume",
                f"Loading {run_config.train.resume_artifact} checkpoint",
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
        _report_startup(
            startup_reporter,
            "startup_prepare",
            "Saving frozen run config",
        )
        save_train_run_config(config=run_config, run_dir=run_paths.run_dir)
        model.set_logger(
            build_tensorboard_logger(
                run_paths,
                step_offset=run_config.train.tensorboard_step_offset,
            )
        )
        _report_startup(
            startup_reporter,
            "startup_prepare",
            "TensorBoard logger ready",
        )
        print_training_startup(
            model=model,
            train_env=train_env,
            config=run_config,
            run_paths=run_paths,
        )
        if run_config.train.save_latest_checkpoint:
            _report_startup(
                startup_reporter,
                "startup_checkpoint",
                "Writing initial latest checkpoint",
            )
            save_latest_artifacts(
                model,
                run_paths,
                policy_metadata=current_policy_artifact_metadata(
                    train_env,
                    model,
                    lineage_step_offset=run_config.train.tensorboard_step_offset,
                ),
            )
        callbacks = build_callbacks(
            env_config=run_config.env,
            train_config=run_config.train,
            curriculum_config=run_config.curriculum,
            run_paths=run_paths,
            initial_curriculum_stage_index=initial_curriculum_stage_index,
            extra_callbacks=extra_callbacks,
        )
        masking_required = training_requires_action_masks(run_config)
        _report_startup(
            startup_reporter,
            "startup_training",
            "Starting training loop",
        )
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
            if model.num_timesteps > 0 and run_config.train.save_latest_checkpoint:
                save_latest_artifacts(
                    model,
                    run_paths,
                    policy_metadata=current_policy_artifact_metadata(
                        train_env,
                        model,
                        lineage_step_offset=run_config.train.tensorboard_step_offset,
                    ),
                )
            raise
        save_artifacts_atomically(
            model=model,
            model_path=run_paths.final_model_path,
            policy_path=run_paths.final_policy_path,
            policy_metadata=current_policy_artifact_metadata(
                train_env,
                model,
                lineage_step_offset=run_config.train.tensorboard_step_offset,
            ),
        )
        if run_config.train.save_latest_checkpoint:
            save_latest_artifacts(
                model,
                run_paths,
                policy_metadata=current_policy_artifact_metadata(
                    train_env,
                    model,
                    lineage_step_offset=run_config.train.tensorboard_step_offset,
                ),
            )
    except Exception:
        cleanup_failed_run(
            run_paths,
            model,
            preserve_run_dir=config.train.explicit_run_dir is not None,
        )
        raise
    finally:
        if train_env is not None:
            train_env.close()


def _report_startup(
    startup_reporter: Callable[[str, str], None] | None,
    kind: str,
    message: str,
) -> None:
    if startup_reporter is not None:
        startup_reporter(kind, message)


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
    model: object,
    total_timesteps: int,
    callback: object,
    use_masking: bool,
    progress_bar: bool,
    reset_num_timesteps: bool,
) -> None:
    """Train one configured SB3 model with the shared learn kwargs."""

    from sb3_contrib import MaskablePPO

    if isinstance(model, MaskablePPO):
        _call_learn(
            model,
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
        _call_learn(
            model,
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
        _call_learn(
            model,
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
        _call_learn(
            model,
            total_timesteps=total_timesteps,
            callback=callback,
            use_masking=use_masking,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        return

    raise TypeError(f"Unsupported training model type: {type(model).__name__}")


def _call_learn(model: object, **kwargs: object) -> None:
    # SB3 and sb3x expose different typed learn signatures; runtime checks above
    # select the compatible kwargs before crossing that dynamic boundary.
    learn = attrgetter("learn")(model)
    learn(**kwargs)
