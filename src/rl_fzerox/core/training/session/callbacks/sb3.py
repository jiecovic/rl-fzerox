# src/rl_fzerox/core/training/session/callbacks/sb3.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.artifacts import (
    current_policy_artifact_metadata,
    save_artifacts_atomically,
    save_recent_checkpoint_artifacts,
    trim_recent_checkpoint_artifacts,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController

from .checkpoints import CheckpointPolicy, resolve_checkpoint_policy
from .metrics import RolloutInfoAccumulator, episode_dicts, info_sequence
from .track_sampling import (
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimePersistence,
    TrackSamplingRuntimeState,
    XCupRotationManager,
    file_track_sampling_runtime_persistence,
    replace_runtime_generation,
)
from .tuning import apply_stage_train_overrides, record_stage_train_overrides


def build_callbacks(
    *,
    env_config: EnvConfig | None = None,
    train_app_config: TrainAppConfig | None = None,
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
    run_paths: RunPaths,
    initial_curriculum_stage_index: int | None = None,
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

    class InfoLoggingCallback(BaseCallback):
        """Log rollout-aggregated state means and episode outcomes."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._rollout_info = RolloutInfoAccumulator()

        def _on_rollout_start(self) -> None:
            self._rollout_info = RolloutInfoAccumulator()

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            self._rollout_info.add_infos(infos)
            return True

        def _on_rollout_end(self) -> None:
            self._rollout_info.record_to(self.logger)

    class RollingArtifactCallback(BaseCallback):
        """Maintain rolling latest and best training artifacts."""

        def __init__(self, *, policy: CheckpointPolicy, run_paths: RunPaths) -> None:
            super().__init__(verbose=0)
            self._policy = policy
            self._run_paths = run_paths
            self._best_episode_return: float | None = None
            self._rollout_count = 0

        def _save_latest(self) -> None:
            save_artifacts_atomically(
                model=self.model,
                model_path=self._run_paths.latest_model_path,
                policy_path=self._run_paths.latest_policy_path,
                policy_metadata=current_policy_artifact_metadata(
                    self.training_env,
                    self.model,
                    lineage_step_offset=train_config.tensorboard_step_offset,
                ),
            )

        def _save_recent(self) -> None:
            num_timesteps = getattr(self.model, "num_timesteps", None)
            if not isinstance(num_timesteps, int):
                return
            save_recent_checkpoint_artifacts(
                self.model,
                self._run_paths,
                num_timesteps=num_timesteps,
                policy_metadata=current_policy_artifact_metadata(
                    self.training_env,
                    self.model,
                    lineage_step_offset=train_config.tensorboard_step_offset,
                ),
            )
            trim_recent_checkpoint_artifacts(
                self._run_paths,
                keep_last=self._policy.recent_limit,
            )

        def _save_periodic(self) -> None:
            if self._policy.save_latest:
                self._save_latest()
            if self._policy.save_recent:
                self._save_recent()

        def _save_best(self, episode_return: float) -> None:
            if (
                self._best_episode_return is not None
                and episode_return <= self._best_episode_return
            ):
                return
            self._best_episode_return = episode_return
            save_artifacts_atomically(
                model=self.model,
                model_path=self._run_paths.best_model_path,
                policy_path=self._run_paths.best_policy_path,
                policy_metadata=current_policy_artifact_metadata(
                    self.training_env,
                    self.model,
                    lineage_step_offset=train_config.tensorboard_step_offset,
                ),
            )

        def _on_training_start(self) -> None:
            if self._policy.save_latest:
                self._save_latest()

        def _on_step(self) -> bool:
            if (
                self._policy.step_interval is not None
                and self.n_calls % self._policy.step_interval == 0
            ):
                self._save_periodic()

            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            if not self._policy.save_best:
                return True

            for info in infos:
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if not isinstance(episode, dict):
                    continue
                episode_return = episode.get("r")
                if isinstance(episode_return, int | float):
                    self._save_best(float(episode_return))
            return True

        def _on_rollout_end(self) -> None:
            if self._policy.rollout_interval is None:
                return
            self._rollout_count += 1
            if self._rollout_count % self._policy.rollout_interval == 0:
                self._save_periodic()

    class CurriculumCallback(BaseCallback):
        """Promote curriculum stages and apply their rollout-time overrides."""

        def __init__(
            self,
            curriculum: CurriculumConfig,
            *,
            env_config: EnvConfig | None = None,
            initial_stage_index: int | None = None,
        ) -> None:
            super().__init__(verbose=0)
            self._controller = ActionMaskCurriculumController(
                curriculum,
                env_config=env_config,
                initial_stage_index=initial_stage_index,
            )

        def _on_training_start(self) -> None:
            self._apply_current_stage()

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            promoted_stage = self._controller.record_episodes(episode_dicts(infos))
            if promoted_stage is not None:
                self._apply_current_stage()
            return True

        def _on_rollout_end(self) -> None:
            stage_index = self._controller.stage_index
            self.logger.record(
                "curriculum/stage",
                -1 if stage_index is None else stage_index,
            )
            record_stage_train_overrides(
                logger=self.logger,
                overrides=self._controller.stage_train_overrides,
            )

        def _apply_current_stage(self) -> None:
            stage_index = self._controller.stage_index
            if stage_index is None:
                return
            self.training_env.env_method("set_curriculum_stage", stage_index)
            apply_stage_train_overrides(
                model=self.model,
                overrides=self._controller.stage_train_overrides,
            )

    class StepBalancedTrackSamplingCallback(BaseCallback):
        """Refresh track sampling weights from completed-episode frame counts."""

        def __init__(
            self,
            *,
            controller: StepBalancedTrackSamplingController,
            env_config: EnvConfig,
            curriculum_config: CurriculumConfig,
            rotation_manager: XCupRotationManager | None,
            runtime_persistence: TrackSamplingRuntimePersistence,
        ) -> None:
            super().__init__(verbose=0)
            self._controller = controller
            self._env_config = env_config
            self._curriculum_config = curriculum_config
            self._rotation_manager = rotation_manager
            self._runtime_persistence = runtime_persistence

        def _on_training_start(self) -> None:
            self.training_env.env_method(
                "set_track_sampling_weights",
                self._controller.current_weights(),
            )
            self._runtime_persistence.save(self._controller.runtime_state())

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            episodes = episode_dicts(infos)
            if not episodes:
                return True
            weights = self._controller.record_episodes(episodes)
            runtime_state = self._controller.runtime_state()
            rotation_manager = self._rotation_manager
            rotation_update = (
                None
                if rotation_manager is None
                else rotation_manager.rotate_once(
                    env_config=self._env_config,
                    state=runtime_state,
                )
            )
            if rotation_update is not None:
                runtime_state = replace_runtime_generation(
                    runtime_state,
                    course_key=rotation_update.replaced_course_key,
                    replacement_label=rotation_update.replacement_label,
                    generated_course_slot=rotation_update.generated_course_slot,
                    generated_course_generation=rotation_update.generated_course_generation,
                    generated_entry_id=rotation_update.generated_entry_id,
                    generated_course_id=rotation_update.generated_course_id,
                    generated_course_name=rotation_update.generated_course_name,
                    generated_course_hash=rotation_update.generated_course_hash,
                    generated_course_seed=rotation_update.generated_course_seed,
                    generated_baseline_state_path=rotation_update.generated_baseline_state_path,
                    generated_course_segment_count=(rotation_update.generated_course_segment_count),
                    generated_course_length=rotation_update.generated_course_length,
                )
                self._env_config = rotation_update.env_config
                self.training_env.env_method(
                    "set_track_sampling_config",
                    self._env_config.track_sampling,
                )
                self._controller = _rebuild_track_sampling_controller(
                    env_config=self._env_config,
                    curriculum_config=self._curriculum_config,
                    restored_state=runtime_state,
                )
                weights = self._controller.current_weights()
                self.training_env.env_method("set_track_sampling_weights", weights)
                if rotation_manager is not None:
                    rotation_manager.commit(rotation_update)
            elif weights is not None:
                self.training_env.env_method("set_track_sampling_weights", weights)
            self._runtime_persistence.save(self._controller.runtime_state())
            return True

        def _on_rollout_end(self) -> None:
            for key, value in self._controller.log_values().items():
                self.logger.record(key, value)

    checkpoint_policy = resolve_checkpoint_policy(train_config)
    callbacks: list[BaseCallback] = [
        RollingArtifactCallback(
            policy=checkpoint_policy,
            run_paths=run_paths,
        ),
        InfoLoggingCallback(),
    ]
    if env_config is not None:
        runtime_persistence = track_sampling_runtime_persistence
        if runtime_persistence is None:
            runtime_persistence = file_track_sampling_runtime_persistence(
                run_paths.track_sampling_state_path
            )
        track_balance_controller = StepBalancedTrackSamplingController.from_configs(
            env_config=env_config,
            curriculum_config=curriculum_config,
            restored_state=runtime_persistence.load(),
        )
        if track_balance_controller is not None:
            callbacks.append(
                StepBalancedTrackSamplingCallback(
                    controller=track_balance_controller,
                    env_config=env_config,
                    curriculum_config=curriculum_config,
                    rotation_manager=_x_cup_rotation_manager(
                        train_app_config=train_app_config,
                        run_paths=run_paths,
                        persist_manifest_on_commit=track_sampling_runtime_persistence is None,
                    ),
                    runtime_persistence=runtime_persistence,
                )
            )
    if curriculum_config.enabled:
        callbacks.append(
            CurriculumCallback(
                curriculum_config,
                env_config=env_config,
                initial_stage_index=initial_curriculum_stage_index,
            )
        )
    callbacks.extend(callback for callback in extra_callbacks if isinstance(callback, BaseCallback))
    return CallbackList(callbacks)


def _rebuild_track_sampling_controller(
    *,
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
    restored_state: TrackSamplingRuntimeState,
) -> StepBalancedTrackSamplingController:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=env_config,
        curriculum_config=curriculum_config,
        restored_state=restored_state,
    )
    if controller is None:
        raise RuntimeError("X Cup rotation removed the dynamic track-sampling controller")
    return controller


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
