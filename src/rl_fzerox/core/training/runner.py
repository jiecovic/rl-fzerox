# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import attrgetter

from rl_fzerox.core.engine_tuning import EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.runtime_spec.x_cup_slots import generated_x_cup_slots_from_track_sampling
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
    load_engine_tuning_checkpoint_state,
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
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimePersistence,
    materialized_track_sampling_artifacts,
)


def run_training(
    config: TrainAppConfig,
    *,
    extra_callbacks: Sequence[object] = (),
    extra_callback_factories: Sequence[
        Callable[[EngineTuningTrainingController | None], object]
    ] = (),
    startup_reporter: Callable[[str, str], None] | None = None,
    track_sampling_runtime_persistence: TrackSamplingRuntimePersistence | None = None,
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
        if track_sampling_runtime_persistence is not None:
            publish_artifacts = track_sampling_runtime_persistence.replace_materialized_artifacts
            if publish_artifacts is not None:
                publish_artifacts(
                    materialized_track_sampling_artifacts(run_config.env.track_sampling)
                )
            publish_generated_slots = (
                track_sampling_runtime_persistence.replace_generated_x_cup_slots
            )
            if publish_generated_slots is not None:
                publish_generated_slots(
                    generated_x_cup_slots_from_track_sampling(run_config.env.track_sampling)
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
            env_config=run_config.env,
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
            policy_config=run_config.policy,
        )
        initial_curriculum_stage_index = _resume_curriculum_stage_index(run_config)
        initial_engine_tuning_state = _resume_engine_tuning_state(run_config)
        if initial_curriculum_stage_index is not None:
            train_env.env_method(
                "sync_checkpoint_curriculum_stage",
                initial_curriculum_stage_index,
            )
        engine_tuning_controller = _engine_tuning_controller(
            run_config,
            state=initial_engine_tuning_state,
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
                engine_tuning_state=_engine_tuning_state(engine_tuning_controller),
                policy_metadata=current_policy_artifact_metadata(
                    train_env,
                    model,
                    lineage_step_offset=run_config.train.tensorboard_step_offset,
                ),
            )
        callbacks = build_callbacks(
            env_config=run_config.env,
            train_app_config=run_config,
            train_config=run_config.train,
            curriculum_config=run_config.curriculum,
            run_paths=run_paths,
            initial_curriculum_stage_index=initial_curriculum_stage_index,
            initial_engine_tuning_state=initial_engine_tuning_state,
            engine_tuning_controller=engine_tuning_controller,
            track_sampling_runtime_persistence=track_sampling_runtime_persistence,
            extra_callbacks=(
                *extra_callbacks,
                *(
                    factory(engine_tuning_controller)
                    for factory in extra_callback_factories
                ),
            ),
        )
        masking_required = training_requires_action_masks(run_config)
        _report_startup(
            startup_reporter,
            "startup_training",
            "Starting training loop",
        )
        reset_num_timesteps = run_config.train.resume_mode != "full_model"
        learn_total_timesteps = _learn_total_timesteps(
            model=model,
            configured_total_timesteps=run_config.train.total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
        )
        try:
            if learn_total_timesteps > 0:
                _learn_model(
                    model=model,
                    total_timesteps=learn_total_timesteps,
                    callback=callbacks,
                    use_masking=masking_required,
                    progress_bar=True,
                    reset_num_timesteps=reset_num_timesteps,
                )
        except Exception:
            if model.num_timesteps > 0 and run_config.train.save_latest_checkpoint:
                save_latest_artifacts(
                    model,
                    run_paths,
                    engine_tuning_state=_engine_tuning_state(engine_tuning_controller),
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
            engine_tuning_state=_engine_tuning_state(engine_tuning_controller),
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
                engine_tuning_state=_engine_tuning_state(engine_tuning_controller),
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


def _resume_engine_tuning_state(config: TrainAppConfig) -> EngineTuningRuntimeState | None:
    if not config.env.track_sampling.engine_tuning.enabled:
        return None
    if config.train.resume_run_dir is None:
        return None
    policy_path = resolve_policy_artifact_path(
        config.train.resume_run_dir,
        artifact=config.train.resume_artifact,
    )
    return load_engine_tuning_checkpoint_state(policy_path)


def _engine_tuning_controller(
    config: TrainAppConfig,
    *,
    state: EngineTuningRuntimeState | None,
) -> EngineTuningTrainingController | None:
    if not config.env.track_sampling.engine_tuning.enabled:
        return None
    return EngineTuningTrainingController(config.env.track_sampling.engine_tuning, state=state)


def _engine_tuning_state(
    controller: EngineTuningTrainingController | None,
) -> EngineTuningRuntimeState | None:
    return None if controller is None else controller.runtime_state


def _learn_total_timesteps(
    *,
    model: object,
    configured_total_timesteps: int,
    reset_num_timesteps: bool,
) -> int:
    """Return the SB3 learn target for this process invocation.

    With `reset_num_timesteps=False`, SB3 treats `total_timesteps` as steps to
    add on top of the loaded model counter. Manager resumes configure an
    absolute run target, so they must pass only the remaining local steps.
    """

    if reset_num_timesteps:
        return configured_total_timesteps
    current_num_timesteps = _model_num_timesteps(model)
    return max(0, configured_total_timesteps - current_num_timesteps)


def _model_num_timesteps(model: object) -> int:
    value = getattr(model, "num_timesteps", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


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
