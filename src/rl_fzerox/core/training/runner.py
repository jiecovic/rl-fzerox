# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from operator import attrgetter
from typing import Protocol

from rl_fzerox.core.engine_tuning import EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.runtime_spec.schema import EnvConfig, TrainAppConfig, TrainConfig
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
from rl_fzerox.core.training.runs.paths import RunPaths
from rl_fzerox.core.training.session import (
    build_callbacks,
    build_tensorboard_logger,
    build_training_env,
    build_training_model,
    cleanup_failed_run,
    current_policy_artifact_metadata,
    load_engine_tuning_checkpoint_state,
    maybe_resume_training_model,
    print_training_startup,
    resolve_train_run_config,
    save_artifacts_atomically,
    save_latest_artifacts,
    training_requires_action_masks,
    validate_training_algorithm_config,
    validate_training_baseline_state,
)
from rl_fzerox.core.training.session.artifacts import (
    ModelSaveable,
    PolicyArtifactMetadata,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimePersistence,
    materialized_track_sampling_artifacts,
)


class _TrainingEnv(Protocol):
    def close(self) -> object: ...


class _TrainingModel(ModelSaveable, Protocol):
    num_timesteps: int

    def set_logger(self, logger: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _EngineTuningSession:
    initial_state: EngineTuningRuntimeState | None
    controller: EngineTuningTrainingController | None


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
    run_paths = _resolve_run_paths(config)
    _report_startup(
        startup_reporter,
        "startup_prepare",
        f"Using run directory {run_paths.run_dir}",
    )
    validate_training_algorithm_config(config)
    train_env: _TrainingEnv | None = None
    model: _TrainingModel | None = None

    try:
        run_config = _prepare_run_config(
            config=config,
            run_paths=run_paths,
            startup_reporter=startup_reporter,
            track_sampling_runtime_persistence=track_sampling_runtime_persistence,
        )
        train_env = _build_training_environment(
            run_config=run_config,
            run_paths=run_paths,
            startup_reporter=startup_reporter,
        )
        model = _build_or_resume_training_model(
            run_config=run_config,
            train_env=train_env,
            startup_reporter=startup_reporter,
        )
        engine_tuning = _build_engine_tuning_session(run_config)
        _prepare_training_outputs(
            run_config=run_config,
            run_paths=run_paths,
            train_env=train_env,
            model=model,
            engine_tuning=engine_tuning,
            startup_reporter=startup_reporter,
        )
        callbacks = _build_training_callbacks(
            env_config=run_config.env,
            train_app_config=run_config,
            train_config=run_config.train,
            run_paths=run_paths,
            engine_tuning=engine_tuning,
            track_sampling_runtime_persistence=track_sampling_runtime_persistence,
            extra_callbacks=(
                *extra_callbacks,
                *(factory(engine_tuning.controller) for factory in extra_callback_factories),
            ),
        )
        masking_required = training_requires_action_masks(run_config)
        _report_startup(
            startup_reporter,
            "startup_training",
            "Starting training loop",
        )
        _train_and_save(
            run_config=run_config,
            run_paths=run_paths,
            model=model,
            callbacks=callbacks,
            masking_required=masking_required,
            engine_tuning=engine_tuning,
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


def _resolve_run_paths(config: TrainAppConfig) -> RunPaths:
    if config.train.continue_run_dir is not None:
        return continue_run_paths(config.train.continue_run_dir)
    if config.train.explicit_run_dir is not None:
        return explicit_run_paths(config.train.explicit_run_dir)
    return reserve_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )


def _prepare_run_config(
    *,
    config: TrainAppConfig,
    run_paths: RunPaths,
    startup_reporter: Callable[[str, str], None] | None,
    track_sampling_runtime_persistence: TrackSamplingRuntimePersistence | None,
) -> TrainAppConfig:
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
    _publish_initial_track_sampling_state(
        run_config,
        track_sampling_runtime_persistence=track_sampling_runtime_persistence,
    )
    validate_training_baseline_state(run_config)
    return run_config


def _publish_initial_track_sampling_state(
    run_config: TrainAppConfig,
    *,
    track_sampling_runtime_persistence: TrackSamplingRuntimePersistence | None,
) -> None:
    if track_sampling_runtime_persistence is None:
        return

    publish_artifacts = track_sampling_runtime_persistence.replace_materialized_artifacts
    if publish_artifacts is not None:
        publish_artifacts(materialized_track_sampling_artifacts(run_config.env.track_sampling))

    publish_generated_slots = track_sampling_runtime_persistence.replace_generated_x_cup_slots
    if publish_generated_slots is not None:
        publish_generated_slots(
            generated_x_cup_slots_from_track_sampling(run_config.env.track_sampling)
        )


def _build_training_environment(
    *,
    run_config: TrainAppConfig,
    run_paths: RunPaths,
    startup_reporter: Callable[[str, str], None] | None,
) -> _TrainingEnv:
    _report_startup(
        startup_reporter,
        "startup_prepare",
        "Building training environments",
    )
    return build_training_env(run_config, run_paths)


def _build_or_resume_training_model(
    *,
    run_config: TrainAppConfig,
    train_env: _TrainingEnv,
    startup_reporter: Callable[[str, str], None] | None,
) -> _TrainingModel:
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
    return maybe_resume_training_model(
        model=model,
        train_env=train_env,
        train_config=run_config.train,
        policy_config=run_config.policy,
    )


def _build_engine_tuning_session(config: TrainAppConfig) -> _EngineTuningSession:
    initial_state = _resume_engine_tuning_state(config)
    return _EngineTuningSession(
        initial_state=initial_state,
        controller=_engine_tuning_controller(config, state=initial_state),
    )


def _prepare_training_outputs(
    *,
    run_config: TrainAppConfig,
    run_paths: RunPaths,
    train_env: _TrainingEnv,
    model: _TrainingModel,
    engine_tuning: _EngineTuningSession,
    startup_reporter: Callable[[str, str], None] | None,
) -> None:
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
        _save_latest_checkpoint(
            run_config=run_config,
            run_paths=run_paths,
            model=model,
            engine_tuning=engine_tuning,
        )


def _build_training_callbacks(
    *,
    env_config: EnvConfig,
    train_app_config: TrainAppConfig,
    train_config: TrainConfig,
    run_paths: RunPaths,
    engine_tuning: _EngineTuningSession,
    track_sampling_runtime_persistence: TrackSamplingRuntimePersistence | None,
    extra_callbacks: Sequence[object],
) -> object:
    return build_callbacks(
        env_config=env_config,
        train_app_config=train_app_config,
        train_config=train_config,
        run_paths=run_paths,
        initial_engine_tuning_state=engine_tuning.initial_state,
        engine_tuning_controller=engine_tuning.controller,
        track_sampling_runtime_persistence=track_sampling_runtime_persistence,
        extra_callbacks=extra_callbacks,
    )


def _train_and_save(
    *,
    run_config: TrainAppConfig,
    run_paths: RunPaths,
    model: _TrainingModel,
    callbacks: object,
    masking_required: bool,
    engine_tuning: _EngineTuningSession,
) -> None:
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
            _save_latest_checkpoint(
                run_config=run_config,
                run_paths=run_paths,
                model=model,
                engine_tuning=engine_tuning,
            )
        raise

    save_artifacts_atomically(
        model=model,
        model_path=run_paths.final_model_path,
        policy_path=run_paths.final_policy_path,
        engine_tuning_state=_engine_tuning_state(engine_tuning.controller),
        policy_metadata=_policy_artifact_metadata(run_config, model),
    )
    if run_config.train.save_latest_checkpoint:
        _save_latest_checkpoint(
            run_config=run_config,
            run_paths=run_paths,
            model=model,
            engine_tuning=engine_tuning,
        )


def _save_latest_checkpoint(
    *,
    run_config: TrainAppConfig,
    run_paths: RunPaths,
    model: _TrainingModel,
    engine_tuning: _EngineTuningSession,
) -> None:
    save_latest_artifacts(
        model,
        run_paths,
        engine_tuning_state=_engine_tuning_state(engine_tuning.controller),
        policy_metadata=_policy_artifact_metadata(run_config, model),
    )


def _policy_artifact_metadata(
    run_config: TrainAppConfig,
    model: _TrainingModel,
) -> PolicyArtifactMetadata:
    return current_policy_artifact_metadata(
        model,
        lineage_step_offset=run_config.train.tensorboard_step_offset,
    )


def _report_startup(
    startup_reporter: Callable[[str, str], None] | None,
    kind: str,
    message: str,
) -> None:
    if startup_reporter is not None:
        startup_reporter(kind, message)


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
