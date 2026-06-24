# src/rl_fzerox/core/training/session/callbacks/sb3/track_sampling.py
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrackSamplingConfig,
    TrainConfig,
)
from rl_fzerox.core.training.session.callbacks.metrics import episode_dicts, info_sequence
from rl_fzerox.core.training.session.callbacks.sb3.env_adapter import (
    TrainingEnvAdapter,
    training_env_adapter,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    DEFICIT_QUEUE_SETTINGS,
    DeficitBudgetTrackSamplingController,
    StepBalancedTrackSamplingController,
    TrackSamplingAltBaseline,
    TrackSamplingRuntimePersistence,
    TrackSamplingRuntimeState,
    XCupRotationManager,
    XCupRotationUpdate,
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


class _TrackSamplingRuntimeCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_config: EnvConfig,
        rotation_manager: XCupRotationManager | None,
        runtime_persistence: TrackSamplingRuntimePersistence,
        alt_baseline_projection: AltBaselineProjectionState | None,
    ) -> None:
        super().__init__(verbose=0)
        self._env_config = env_config
        self._rotation_manager = rotation_manager
        self._runtime_persistence = runtime_persistence
        self._alt_baseline_projection = alt_baseline_projection
        self._runtime_state_dirty = False

    def _controller_runtime_state(self) -> TrackSamplingRuntimeState:
        raise NotImplementedError

    def _on_training_end(self) -> None:
        self._save_runtime_state()

    def _save_runtime_state(self) -> None:
        self._runtime_persistence.save(self._controller_runtime_state())
        self._runtime_state_dirty = False

    def _env(self) -> TrainingEnvAdapter:
        return training_env_adapter(self.training_env)

    def _publish_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        if self._alt_baseline_projection is not None:
            config = self._alt_baseline_projection.project_fresh(config)
        self._env().set_track_sampling_config(config)

    def _record_log_values(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            self.logger.record(key, value)

    def _begin_x_cup_rotation(
        self,
        runtime_state: TrackSamplingRuntimeState,
    ) -> tuple[TrackSamplingRuntimeState, XCupRotationUpdate] | None:
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
            return None
        replaced_state = replace_runtime_generation(
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
        return replaced_state, rotation_update

    def _finish_x_cup_rotation(self, rotation_update: XCupRotationUpdate) -> None:
        persist_artifacts = self._runtime_persistence.replace_materialized_artifacts
        if persist_artifacts is not None:
            persist_artifacts(rotation_update.materialized_artifacts)
        persist_slots = self._runtime_persistence.replace_generated_x_cup_slots
        if persist_slots is not None:
            persist_slots(rotation_update.generated_x_cup_slots)
        if self._rotation_manager is not None:
            self._rotation_manager.commit(rotation_update)
        self._save_runtime_state()


class StepBalancedTrackSamplingCallback(_TrackSamplingRuntimeCallback):
    """Refresh track sampling weights from completed-episode frame counts."""

    def __init__(
        self,
        *,
        controller: StepBalancedTrackSamplingController,
        env_config: EnvConfig,
        rotation_manager: XCupRotationManager | None,
        runtime_persistence: TrackSamplingRuntimePersistence,
        alt_baseline_projection: AltBaselineProjectionState | None,
    ) -> None:
        super().__init__(
            env_config=env_config,
            rotation_manager=rotation_manager,
            runtime_persistence=runtime_persistence,
            alt_baseline_projection=alt_baseline_projection,
        )
        self._controller = controller

    def _controller_runtime_state(self) -> TrackSamplingRuntimeState:
        return self._controller.runtime_state()

    def _on_training_start(self) -> None:
        self._env().set_track_sampling_weights(self._controller.current_weights())
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
        rotation = self._begin_x_cup_rotation(runtime_state)
        if rotation is not None:
            runtime_state, rotation_update = rotation
            self._controller = _rebuild_track_sampling_controller(
                env_config=self._env_config,
                restored_state=runtime_state,
            )
            weights = self._controller.current_weights()
            self._env().set_track_sampling_weights(weights)
            self._finish_x_cup_rotation(rotation_update)
        elif weights is not None:
            self._env().set_track_sampling_weights(weights)
            self._save_runtime_state()
        return True

    def _on_rollout_end(self) -> None:
        self._record_log_values(self._controller.log_values())
        if self._runtime_state_dirty:
            self._save_runtime_state()


class DeficitBudgetTrackSamplingCallback(_TrackSamplingRuntimeCallback):
    """Schedule reset queues from deterministic per-course step deficits."""

    def __init__(
        self,
        *,
        controller: DeficitBudgetTrackSamplingController,
        env_config: EnvConfig,
        rotation_manager: XCupRotationManager | None,
        runtime_persistence: TrackSamplingRuntimePersistence,
        alt_baseline_projection: AltBaselineProjectionState | None,
        train_config: TrainConfig,
    ) -> None:
        super().__init__(
            env_config=env_config,
            rotation_manager=rotation_manager,
            runtime_persistence=runtime_persistence,
            alt_baseline_projection=alt_baseline_projection,
        )
        self._controller = controller
        self._train_config = train_config
        self._rollout_budget_bootstrapped = False

    def _controller_runtime_state(self) -> TrackSamplingRuntimeState:
        return self._controller.runtime_state()

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
        self._record_log_values(self._controller.log_values())
        if self._runtime_state_dirty:
            self._save_runtime_state()

    def _maybe_rotate_x_cup(self) -> bool:
        runtime_state = self._controller.runtime_state()
        rotation = self._begin_x_cup_rotation(runtime_state)
        if rotation is None:
            return False
        runtime_state, rotation_update = rotation
        self._controller = _rebuild_deficit_budget_track_sampling_controller(
            env_config=self._env_config,
            restored_state=runtime_state,
        )
        self._replace_env_queues()
        self._finish_x_cup_rotation(rotation_update)
        return True

    def _refill_env_queues(self) -> None:
        queue_lengths = self._env().track_sampling_reset_queue_lengths()
        self._extend_env_queues(
            self._controller.refill_queues(
                queue_lengths,
                fallback_assignment_steps=self._train_config.n_steps,
            )
        )

    def _replace_env_queues(self) -> None:
        self._env().clear_track_sampling_reset_queue()
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
            self._env().extend_track_sampling_reset_queue(
                env_index=env_index,
                queued_resets=queued_resets,
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
        training_env_adapter(self.training_env).set_track_sampling_config(next_track_sampling)


def _rebuild_track_sampling_controller(
    *,
    env_config: EnvConfig,
    restored_state: TrackSamplingRuntimeState,
) -> StepBalancedTrackSamplingController:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=env_config,
        restored_state=restored_state,
    )
    if controller is None:
        raise RuntimeError("X Cup rotation removed the dynamic track-sampling controller")
    return controller


def _rebuild_deficit_budget_track_sampling_controller(
    *,
    env_config: EnvConfig,
    restored_state: TrackSamplingRuntimeState,
) -> DeficitBudgetTrackSamplingController:
    controller = DeficitBudgetTrackSamplingController.from_configs(
        env_config=env_config,
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
