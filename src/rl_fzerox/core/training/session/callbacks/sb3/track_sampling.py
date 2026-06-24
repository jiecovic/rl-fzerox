# src/rl_fzerox/core/training/session/callbacks/sb3/track_sampling.py
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrainConfig,
)
from rl_fzerox.core.runtime_spec.x_cup_slots import GeneratedXCupSlot
from rl_fzerox.core.training.session.callbacks.metrics import episode_dicts, info_sequence
from rl_fzerox.core.training.session.callbacks.track_sampling import (
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
    replace_runtime_generation,
    strip_alt_baselines,
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
                generated_course_segment_count=rotation_update.generated_course_segment_count,
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
        train_config: TrainConfig,
    ) -> None:
        super().__init__(verbose=0)
        self._controller = controller
        self._env_config = env_config
        self._curriculum_config = curriculum_config
        self._rotation_manager = rotation_manager
        self._runtime_persistence = runtime_persistence
        self._alt_baseline_projection = alt_baseline_projection
        self._train_config = train_config
        self._runtime_state_dirty = False
        self._rollout_budget_bootstrapped = False

    def _on_training_start(self) -> None:
        self._add_rollout_budget()
        self._rollout_budget_bootstrapped = True
        self._extend_env_queues(
            self._controller.initial_queues(
                num_envs=self._train_config.num_envs,
                queue_length=DEFICIT_QUEUE_SETTINGS.initial_queue_length,
                fallback_assignment_steps=self._train_config.n_steps,
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
            total_steps=self._train_config.num_envs * self._train_config.n_steps,
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
                fallback_assignment_steps=self._train_config.n_steps,
            )
        )

    def _replace_env_queues(self) -> None:
        self.training_env.env_method("clear_track_sampling_reset_queue")
        self._controller.clear_reserved_assignments()
        self._extend_env_queues(
            self._controller.initial_queues(
                num_envs=self._train_config.num_envs,
                queue_length=DEFICIT_QUEUE_SETTINGS.initial_queue_length,
                fallback_assignment_steps=self._train_config.n_steps,
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
