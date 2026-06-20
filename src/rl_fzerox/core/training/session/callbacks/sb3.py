# src/rl_fzerox/core/training/session/callbacks/sb3.py
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from rl_fzerox.core.engine_tuning import EngineTuningContext, EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.envs.engine.reset.track_sampling import (
    TrackSamplingQueuedReset,
    engine_tuning_context_for_entry,
)
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
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
    DEFICIT_QUEUE_SETTINGS,
    DeficitBudgetTrackSamplingController,
    StepBalancedTrackSamplingController,
    TrackSamplingAltBaseline,
    TrackSamplingMaterializedArtifact,
    TrackSamplingRuntimePersistence,
    TrackSamplingRuntimeState,
    XCupRotationManager,
    alt_baseline_signature,
    apply_alt_baselines_to_track_sampling,
    file_track_sampling_runtime_persistence,
    replace_runtime_generation,
    strip_alt_baselines,
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

        def __init__(
            self,
            *,
            engine_tuning_controller: EngineTuningTrainingController | None,
            policy: CheckpointPolicy,
            run_paths: RunPaths,
        ) -> None:
            super().__init__(verbose=0)
            self._engine_tuning_controller = engine_tuning_controller
            self._policy = policy
            self._run_paths = run_paths
            self._best_episode_return: float | None = None
            self._rollout_count = 0

        def _save_latest(self) -> None:
            save_artifacts_atomically(
                model=self.model,
                model_path=self._run_paths.latest_model_path,
                policy_path=self._run_paths.latest_policy_path,
                engine_tuning_state=self._engine_tuning_state(),
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
                engine_tuning_state=self._engine_tuning_state(),
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
                engine_tuning_state=self._engine_tuning_state(),
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

        def _engine_tuning_state(self) -> EngineTuningRuntimeState | None:
            if self._engine_tuning_controller is None:
                return None
            return self._engine_tuning_controller.runtime_state

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

    class AltBaselineProjectionState:
        """Project manager-owned alt baselines onto the latest base reset config."""

        def __init__(
            self,
            *,
            env_config: EnvConfig,
            load_alt_baselines: Callable[[], tuple[TrackSamplingAltBaseline, ...]],
        ) -> None:
            self._base_track_sampling = strip_alt_baselines(env_config.track_sampling)
            self._load_alt_baselines = load_alt_baselines
            self._last_baseline_signature: tuple[tuple[object, ...], ...] | None = None
            self._last_base_signature: tuple[tuple[object, ...], ...] | None = None

        def set_base_track_sampling(self, track_sampling: TrackSamplingConfig) -> None:
            self._base_track_sampling = strip_alt_baselines(track_sampling)

        def refreshed_track_sampling(self) -> TrackSamplingConfig | None:
            baselines = self._load_alt_baselines()
            baseline_signature = alt_baseline_signature(baselines)
            base_signature = _track_sampling_config_signature(self._base_track_sampling)
            if (
                baseline_signature == self._last_baseline_signature
                and base_signature == self._last_base_signature
            ):
                return None
            self._last_baseline_signature = baseline_signature
            self._last_base_signature = base_signature
            return apply_alt_baselines_to_track_sampling(
                self._base_track_sampling,
                baselines,
            )

        def project_fresh(self, track_sampling: TrackSamplingConfig) -> TrackSamplingConfig:
            self.set_base_track_sampling(track_sampling)
            baselines = self._load_alt_baselines()
            self._last_baseline_signature = alt_baseline_signature(baselines)
            self._last_base_signature = _track_sampling_config_signature(self._base_track_sampling)
            return apply_alt_baselines_to_track_sampling(
                self._base_track_sampling,
                baselines,
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
            alt_baseline_projection: AltBaselineProjectionState | None,
        ) -> None:
            super().__init__(verbose=0)
            self._controller = controller
            self._env_config = env_config
            self._curriculum_config = curriculum_config
            self._rotation_manager = rotation_manager
            self._runtime_persistence = runtime_persistence
            self._alt_baseline_projection = alt_baseline_projection
            self._runtime_state_dirty = False

        def _on_training_start(self) -> None:
            self.training_env.env_method(
                "set_track_sampling_weights",
                self._controller.current_weights(),
            )
            self._save_runtime_state()

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            episodes = episode_dicts(infos)
            if not episodes:
                return True
            weights = self._controller.record_episodes(episodes)
            self._runtime_state_dirty = True
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
                    generated_course_id=rotation_update.generated_course_id,
                    generated_course_name=rotation_update.generated_course_name,
                    generated_course_hash=rotation_update.generated_course_hash,
                    generated_course_seed=rotation_update.generated_course_seed,
                    generated_course_segment_count=(rotation_update.generated_course_segment_count),
                    generated_course_length=rotation_update.generated_course_length,
                )
                self._env_config = rotation_update.env_config
                self._publish_track_sampling_config(self._env_config.track_sampling)
                self._controller = _rebuild_track_sampling_controller(
                    env_config=self._env_config,
                    curriculum_config=self._curriculum_config,
                    restored_state=runtime_state,
                )
                weights = self._controller.current_weights()
                self.training_env.env_method("set_track_sampling_weights", weights)
                self._save_materialized_artifacts(rotation_update.materialized_artifacts)
                self._save_generated_x_cup_slots(rotation_update.generated_x_cup_slots)
                if rotation_manager is not None:
                    rotation_manager.commit(rotation_update)
                self._save_runtime_state()
            elif weights is not None:
                self.training_env.env_method("set_track_sampling_weights", weights)
                self._save_runtime_state()
            return True

        def _on_rollout_end(self) -> None:
            for key, value in self._controller.log_values().items():
                self.logger.record(key, value)
            if self._runtime_state_dirty:
                self._save_runtime_state()

        def _on_training_end(self) -> None:
            self._save_runtime_state()

        def _save_runtime_state(self) -> None:
            self._runtime_persistence.save(self._controller.runtime_state())
            self._runtime_state_dirty = False

        def _save_materialized_artifacts(
            self,
            artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
        ) -> None:
            persist = self._runtime_persistence.replace_materialized_artifacts
            if persist is not None:
                persist(artifacts)

        def _save_generated_x_cup_slots(
            self,
            slots: tuple[GeneratedXCupSlot, ...],
        ) -> None:
            persist = self._runtime_persistence.replace_generated_x_cup_slots
            if persist is not None:
                persist(slots)

        def _publish_track_sampling_config(self, config: TrackSamplingConfig) -> None:
            if self._alt_baseline_projection is not None:
                config = self._alt_baseline_projection.project_fresh(config)
            self.training_env.env_method("set_track_sampling_config", config)

    class DeficitBudgetTrackSamplingCallback(BaseCallback):
        """Schedule reset queues from deterministic per-course step deficits."""

        def __init__(
            self,
            *,
            controller: DeficitBudgetTrackSamplingController,
            env_config: EnvConfig,
            curriculum_config: CurriculumConfig,
            rotation_manager: XCupRotationManager | None,
            runtime_persistence: TrackSamplingRuntimePersistence,
            alt_baseline_projection: AltBaselineProjectionState | None,
        ) -> None:
            super().__init__(verbose=0)
            self._controller = controller
            self._env_config = env_config
            self._curriculum_config = curriculum_config
            self._rotation_manager = rotation_manager
            self._runtime_persistence = runtime_persistence
            self._alt_baseline_projection = alt_baseline_projection
            self._runtime_state_dirty = False
            self._rollout_budget_bootstrapped = False

        def _on_training_start(self) -> None:
            self._add_rollout_budget()
            self._rollout_budget_bootstrapped = True
            self._extend_env_queues(
                self._controller.initial_queues(
                    num_envs=train_config.num_envs,
                    queue_length=DEFICIT_QUEUE_SETTINGS.initial_queue_length,
                    fallback_assignment_steps=train_config.n_steps,
                )
            )
            self._save_runtime_state()

        def _on_rollout_start(self) -> None:
            if self._rollout_budget_bootstrapped:
                self._rollout_budget_bootstrapped = False
            else:
                self._add_rollout_budget()
            self._replace_env_queues()

        def _add_rollout_budget(self) -> None:
            self._controller.add_rollout_budget(
                total_steps=train_config.num_envs * train_config.n_steps,
            )

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True
            self._controller.record_step_infos(infos)
            episodes = episode_dicts(infos)
            if episodes:
                self._controller.record_episodes(episodes)
                self._runtime_state_dirty = True
                rotated = self._maybe_rotate_x_cup()
                if not rotated:
                    self._refill_env_queues()
            return True

        def _on_rollout_end(self) -> None:
            if self._controller.maybe_update_weights():
                self._replace_env_queues()
                self._runtime_state_dirty = True
            for key, value in self._controller.log_values().items():
                self.logger.record(key, value)
            if self._runtime_state_dirty:
                self._save_runtime_state()

        def _on_training_end(self) -> None:
            self._save_runtime_state()

        def _maybe_rotate_x_cup(self) -> bool:
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
            if rotation_update is None:
                return False
            runtime_state = replace_runtime_generation(
                runtime_state,
                course_key=rotation_update.replaced_course_key,
                replacement_label=rotation_update.replacement_label,
                generated_course_slot=rotation_update.generated_course_slot,
                generated_course_generation=rotation_update.generated_course_generation,
                generated_course_id=rotation_update.generated_course_id,
                generated_course_name=rotation_update.generated_course_name,
                generated_course_hash=rotation_update.generated_course_hash,
                generated_course_seed=rotation_update.generated_course_seed,
                generated_course_segment_count=rotation_update.generated_course_segment_count,
                generated_course_length=rotation_update.generated_course_length,
            )
            self._env_config = rotation_update.env_config
            self._publish_track_sampling_config(self._env_config.track_sampling)
            self._controller = _rebuild_deficit_budget_track_sampling_controller(
                env_config=self._env_config,
                curriculum_config=self._curriculum_config,
                restored_state=runtime_state,
            )
            self._replace_env_queues()
            self._save_materialized_artifacts(rotation_update.materialized_artifacts)
            self._save_generated_x_cup_slots(rotation_update.generated_x_cup_slots)
            if rotation_manager is not None:
                rotation_manager.commit(rotation_update)
            self._save_runtime_state()
            return True

        def _refill_env_queues(self) -> None:
            raw_lengths = self.training_env.env_method("track_sampling_reset_queue_length")
            queue_lengths = tuple(
                int(length) if isinstance(length, int | float) else 0 for length in raw_lengths
            )
            self._extend_env_queues(
                self._controller.refill_queues(
                    queue_lengths,
                    fallback_assignment_steps=train_config.n_steps,
                )
            )

        def _replace_env_queues(self) -> None:
            self.training_env.env_method("clear_track_sampling_reset_queue")
            self._controller.clear_reserved_assignments()
            self._extend_env_queues(
                self._controller.initial_queues(
                    num_envs=train_config.num_envs,
                    queue_length=DEFICIT_QUEUE_SETTINGS.initial_queue_length,
                    fallback_assignment_steps=train_config.n_steps,
                )
            )

        def _extend_env_queues(
            self,
            queues: Mapping[int, Sequence[TrackSamplingQueuedReset]],
        ) -> None:
            for env_index, queued_resets in queues.items():
                if not queued_resets:
                    continue
                self.training_env.env_method(
                    "extend_track_sampling_reset_queue",
                    tuple(queued_resets),
                    indices=[env_index],
                )

        def _save_runtime_state(self) -> None:
            self._runtime_persistence.save(self._controller.runtime_state())
            self._runtime_state_dirty = False

        def _save_materialized_artifacts(
            self,
            artifacts: tuple[TrackSamplingMaterializedArtifact, ...],
        ) -> None:
            persist = self._runtime_persistence.replace_materialized_artifacts
            if persist is not None:
                persist(artifacts)

        def _save_generated_x_cup_slots(
            self,
            slots: tuple[GeneratedXCupSlot, ...],
        ) -> None:
            persist = self._runtime_persistence.replace_generated_x_cup_slots
            if persist is not None:
                persist(slots)

        def _publish_track_sampling_config(self, config: TrackSamplingConfig) -> None:
            if self._alt_baseline_projection is not None:
                config = self._alt_baseline_projection.project_fresh(config)
            self.training_env.env_method("set_track_sampling_config", config)

    class EngineTuningCallback(BaseCallback):
        """Update adaptive engine-setting stats from completed episodes."""

        def __init__(
            self,
            *,
            controller: EngineTuningTrainingController,
            contexts: Sequence[EngineTuningContext],
        ) -> None:
            super().__init__(verbose=0)
            self._controller = controller
            self._contexts = tuple(contexts)
            self._sampler_dirty = False

        def _on_training_start(self) -> None:
            self._publish_sampler()

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True
            episodes = episode_dicts(infos)
            if not episodes:
                return True
            if self._controller.record_episodes(episodes):
                self._sampler_dirty = True
            return True

        def _on_rollout_end(self) -> None:
            rollout_changed = self._controller.record_rollout_episodes()
            if self._sampler_dirty or rollout_changed:
                self._publish_sampler()
                self._sampler_dirty = False
            for key, value in self._controller.log_values().items():
                self.logger.record(key, value)

        def _publish_sampler(self) -> None:
            self.training_env.env_method(
                "set_engine_tuning_sampler",
                self._controller.reset_sampler_snapshot(self._contexts),
            )

    class AltBaselineSyncCallback(BaseCallback):
        """Publish manager-owned alt baselines to env workers at rollout boundaries."""

        def __init__(
            self,
            *,
            projection: AltBaselineProjectionState,
        ) -> None:
            super().__init__(verbose=0)
            self._projection = projection

        def _on_training_start(self) -> None:
            self._sync()

        def _on_rollout_start(self) -> None:
            self._sync()

        def _on_step(self) -> bool:
            return True

        def _sync(self) -> None:
            next_track_sampling = self._projection.refreshed_track_sampling()
            if next_track_sampling is None:
                return
            self.training_env.env_method("set_track_sampling_config", next_track_sampling)

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
                contexts=_engine_tuning_contexts(env_config),
            )
        )
    callbacks.extend(
        (
            RollingArtifactCallback(
                engine_tuning_controller=engine_tuning_controller,
                policy=checkpoint_policy,
                run_paths=run_paths,
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
            curriculum_config=curriculum_config,
            restored_state=runtime_persistence.load(),
        )
        if deficit_controller is not None:
            callbacks.append(
                DeficitBudgetTrackSamplingCallback(
                    controller=deficit_controller,
                    env_config=env_config,
                    curriculum_config=curriculum_config,
                    rotation_manager=_x_cup_rotation_manager(
                        train_app_config=train_app_config,
                        run_paths=run_paths,
                        persist_manifest_on_commit=track_sampling_runtime_persistence is None,
                    ),
                    runtime_persistence=runtime_persistence,
                    alt_baseline_projection=alt_baseline_projection,
                )
            )

        if deficit_controller is None:
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
                        alt_baseline_projection=alt_baseline_projection,
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


def _rebuild_deficit_budget_track_sampling_controller(
    *,
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
    restored_state: TrackSamplingRuntimeState,
) -> DeficitBudgetTrackSamplingController:
    controller = DeficitBudgetTrackSamplingController.from_configs(
        env_config=env_config,
        curriculum_config=curriculum_config,
        restored_state=restored_state,
    )
    if controller is None:
        raise RuntimeError("X Cup rotation removed the deficit-budget track-sampling controller")
    return controller


def _engine_tuning_contexts(env_config: EnvConfig) -> tuple[EngineTuningContext, ...]:
    contexts: dict[str, EngineTuningContext] = {}
    for entry in env_config.track_sampling.entries:
        context = engine_tuning_context_for_entry(entry)
        contexts.setdefault(context.key, context)
    return tuple(contexts[key] for key in sorted(contexts))


def _track_sampling_config_signature(config: TrackSamplingConfig) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            entry.id,
            entry.baseline_state_path,
            float(entry.weight),
            entry.course_id,
            entry.runtime_course_key,
            entry.mode,
            entry.gp_difficulty,
            entry.vehicle,
            entry.generated_course_kind,
            entry.generated_course_slot,
            entry.generated_course_generation,
            entry.generated_course_hash,
        )
        for entry in config.entries
    )


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
