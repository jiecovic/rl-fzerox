# src/rl_fzerox/core/training/session/callbacks/sb3/factory.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.core.engine_tuning import EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.callbacks.checkpoints import resolve_checkpoint_policy
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    DeficitBudgetTrackSamplingController,
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimePersistence,
    XCupRotationManager,
    file_track_sampling_runtime_persistence,
)


def build_callbacks(
    *,
    env_config: EnvConfig | None = None,
    train_app_config: TrainAppConfig | None = None,
    train_config: TrainConfig,
    run_paths: RunPaths,
    initial_engine_tuning_state: EngineTuningRuntimeState | None = None,
    engine_tuning_controller: EngineTuningTrainingController | None = None,
    track_sampling_runtime_persistence: TrackSamplingRuntimePersistence | None = None,
    extra_callbacks: Sequence[object] = (),
):
    """Construct the SB3 callback list used during training."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    from rl_fzerox.core.training.session.callbacks.sb3.artifacts import (
        RollingArtifactCallback,
    )
    from rl_fzerox.core.training.session.callbacks.sb3.engine_tuning import (
        EngineTuningCallback,
        engine_tuning_contexts,
    )
    from rl_fzerox.core.training.session.callbacks.sb3.rollout_logging import (
        InfoLoggingCallback,
    )
    from rl_fzerox.core.training.session.callbacks.sb3.track_sampling import (
        AltBaselineProjectionState,
        AltBaselineSyncCallback,
        DeficitBudgetTrackSamplingCallback,
        StepBalancedTrackSamplingCallback,
    )

    checkpoint_policy = resolve_checkpoint_policy(train_config)
    if (
        engine_tuning_controller is None
        and env_config is not None
        and env_config.track_sampling.engine_tuning.enabled
    ):
        engine_tuning_controller = EngineTuningTrainingController(
            env_config.track_sampling.engine_tuning,
            state=initial_engine_tuning_state,
        )
    callbacks: list[BaseCallback] = []
    if engine_tuning_controller is not None and env_config is not None:
        callbacks.append(
            EngineTuningCallback(
                controller=engine_tuning_controller,
                contexts=engine_tuning_contexts(env_config),
            )
        )
    callbacks.extend(
        (
            RollingArtifactCallback(
                engine_tuning_controller=engine_tuning_controller,
                policy=checkpoint_policy,
                run_paths=run_paths,
                lineage_step_offset=train_config.tensorboard_step_offset,
            ),
            InfoLoggingCallback(),
        )
    )
    if env_config is not None:
        runtime_persistence = track_sampling_runtime_persistence
        if runtime_persistence is None:
            runtime_persistence = file_track_sampling_runtime_persistence(
                run_paths.track_sampling_state_path
            )
        alt_baseline_projection: AltBaselineProjectionState | None = None
        if runtime_persistence.load_alt_baselines is not None:
            alt_baseline_projection = AltBaselineProjectionState(
                env_config=env_config,
                load_alt_baselines=runtime_persistence.load_alt_baselines,
            )
            callbacks.append(
                AltBaselineSyncCallback(
                    projection=alt_baseline_projection,
                )
            )
        deficit_controller = DeficitBudgetTrackSamplingController.from_configs(
            env_config=env_config,
            restored_state=runtime_persistence.load(),
        )
        if deficit_controller is not None:
            callbacks.append(
                DeficitBudgetTrackSamplingCallback(
                    controller=deficit_controller,
                    env_config=env_config,
                    rotation_manager=_x_cup_rotation_manager(
                        train_app_config=train_app_config,
                        run_paths=run_paths,
                        persist_manifest_on_commit=track_sampling_runtime_persistence is None,
                    ),
                    runtime_persistence=runtime_persistence,
                    alt_baseline_projection=alt_baseline_projection,
                    train_config=train_config,
                )
            )

        if deficit_controller is None:
            track_balance_controller = StepBalancedTrackSamplingController.from_configs(
                env_config=env_config,
                restored_state=runtime_persistence.load(),
            )
            if track_balance_controller is not None:
                callbacks.append(
                    StepBalancedTrackSamplingCallback(
                        controller=track_balance_controller,
                        env_config=env_config,
                        rotation_manager=_x_cup_rotation_manager(
                            train_app_config=train_app_config,
                            run_paths=run_paths,
                            persist_manifest_on_commit=track_sampling_runtime_persistence is None,
                        ),
                        runtime_persistence=runtime_persistence,
                        alt_baseline_projection=alt_baseline_projection,
                    )
                )
    callbacks.extend(callback for callback in extra_callbacks if isinstance(callback, BaseCallback))
    return CallbackList(callbacks)


def _x_cup_rotation_manager(
    *,
    train_app_config: TrainAppConfig | None,
    run_paths: RunPaths,
    persist_manifest_on_commit: bool,
) -> XCupRotationManager | None:
    if train_app_config is None or not train_app_config.env.track_sampling.x_cup_rotation.enabled:
        return None
    return XCupRotationManager(
        config=train_app_config,
        run_paths=run_paths,
        persist_manifest_on_commit=persist_manifest_on_commit,
    )
