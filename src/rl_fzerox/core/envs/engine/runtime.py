# src/rl_fzerox/core/envs/engine/runtime.py
from __future__ import annotations

from collections.abc import Sequence

from gymnasium import spaces

from fzerox_emulator import ControllerState, EmulatorBackend, RaceControlState, SpinRequest
from fzerox_emulator.arrays import ActionMask, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import (
    ActionValue,
    DecodedAction,
    DiscreteActionDimension,
    ResettableActionAdapter,
    build_action_adapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.rewards import build_reward_tracker
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_vector_or_zeros,
)
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, KNOWN_RENDERERS, RendererName
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)

from .controls import (
    ActionMaskBranches,
    ActionMaskController,
    ActionMaskSnapshot,
    ControlStateTracker,
    action_branch_value_allowed,
    apply_dynamic_control_gates,
    sync_dynamic_action_masks,
)
from .episode import EngineEpisodeState
from .info import (
    backend_step_info,
    set_curriculum_info,
    telemetry_info,
)
from .observation import EngineObservationBuilder
from .reset import EngineResetCoordinator
from .stepping import (
    EngineStepAssembler,
    EnvStepRequest,
    PolicyDriveStep,
    WatchEnvStep,
    policy_drive_info,
    set_episode_boost_pad_info,
)


class FZeroXEnvEngine:
    """Environment step engine around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply reward shaping,
    build policy observations, and assemble Gym-facing info.
    """

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
        env_index: int = 0,
    ) -> None:
        self.backend = backend
        self.config = config
        self._renderer: RendererName = _backend_renderer(backend)
        self._curriculum_config = curriculum_config
        self._action_config = config.action.runtime()
        self._action_adapter = build_action_adapter(self._action_config)
        self._observation_builder = EngineObservationBuilder.from_engine_config(
            backend=backend,
            config=config,
            renderer=self._renderer,
        )
        self._reward_tracker = build_reward_tracker(
            config=reward_config,
            max_episode_steps=config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._action_space = self._action_adapter.action_space
        self._observation_space = self._observation_builder.space
        self._mask_controller = ActionMaskController.from_config(
            adapter=self._action_adapter,
            base_overrides=self._action_config.mask_overrides,
            curriculum_config=curriculum_config,
            boost_unmask_max_speed_kph=self._action_config.boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=self._action_config.lean_unmask_min_speed_kph,
            mask_air_brake_on_ground=self._action_config.mask_air_brake_on_ground,
            pitch_neutral_index=self._action_config.pitch_buckets // 2,
        )
        self._reset_coordinator = EngineResetCoordinator(
            backend=backend,
            config=config,
            curriculum_config=curriculum_config,
            env_index=env_index,
        )
        self._reset_coordinator.set_curriculum_stage(self._mask_controller.stage_index)
        self._control_state = ControlStateTracker(
            lean_mode=self._action_config.lean_mode,
            lean_initial_lockout_frames=self._action_config.lean_initial_lockout_frames,
            boost_decision_interval_frames=self._action_config.boost_decision_interval_frames,
            boost_request_lockout_frames=self._action_config.boost_request_lockout_frames,
            action_history_len=self._observation_builder.action_history_len,
            action_history_controls=self._observation_builder.action_history_controls,
            split_lean_history=self._action_config.split_lean_history,
        )
        self._step_assembler = EngineStepAssembler(
            backend=self.backend,
            config=self.config,
            action_config=self._action_config,
            reward_summary_config=self._reward_summary_config,
            reward_tracker=self._reward_tracker,
            observation_builder=self._observation_builder,
            mask_controller=self._mask_controller,
            control_state=self._control_state,
            renderer=self._renderer,
        )
        self._episode = EngineEpisodeState()

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        return self._action_adapter.action_dimensions

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._episode.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._episode.last_gas_level

    def action_masks(self) -> ActionMask:
        """Return the flattened boolean action mask for the current stage."""

        return self._mask_controller.action_mask()

    def action_mask_branches(self) -> ActionMaskBranches:
        """Return the current action mask grouped by branch label."""

        return self._mask_controller.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        """Return the flat mask and branch view from one mask computation."""

        return self._mask_controller.action_mask_snapshot()

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        self._mask_controller.set_curriculum_stage(stage_index)
        self._reset_coordinator.set_curriculum_stage(self._mask_controller.stage_index)

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        """Align watch-time stage masks with the loaded checkpoint metadata."""

        self._mask_controller.sync_checkpoint_stage(stage_index)
        self._reset_coordinator.set_curriculum_stage(self._mask_controller.stage_index)

    def set_track_sampling_weights(self, weights_by_track_id: dict[str, float]) -> None:
        """Update adaptive reset weights used by step-balanced track sampling."""

        self._reset_coordinator.set_track_sampling_weights(weights_by_track_id)

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        """Replace reset candidates after generated-course rotation."""

        self._reset_coordinator.set_track_sampling_config(config)

    def extend_track_sampling_reset_queue(self, course_ids: Sequence[str]) -> None:
        """Append externally scheduled course ids for deficit-budget resets."""

        self._reset_coordinator.extend_track_sampling_reset_queue(course_ids)

    def track_sampling_reset_queue_length(self) -> int:
        """Return queued deficit-budget reset course count."""

        return self._reset_coordinator.track_sampling_reset_queue_length()

    def set_locked_reset_course(self, course_id: str | None) -> None:
        """Lock subsequent sampled resets to one course for watch/manual inspection."""

        self._reset_coordinator.set_locked_reset_course(course_id)

    def set_sequential_track_sampling(self, enabled: bool) -> None:
        """Use configured track order for watch resets instead of training sampling."""

        self._reset_coordinator.set_sequential_track_sampling(enabled)

    def set_next_sequential_reset_course(self, course_id: str | None) -> None:
        """Align the next sequential watch reset to a specific configured course."""

        self._reset_coordinator.set_next_sequential_course(course_id)

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._mask_controller.stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        return self._mask_controller.stage_name

    def auxiliary_state_targets(self) -> StateVector:
        """Return the current hidden auxiliary-state target vector."""

        return auxiliary_state_target_vector_or_zeros(self._episode.last_telemetry)

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode and return the first policy observation."""

        selected_track = self._reset_coordinator.select_episode_track(seed)
        self._episode.begin_reset(active_track=selected_track)
        reset_result = self._reset_coordinator.reset_race(
            seed=seed,
            selected_track=selected_track,
        )
        info = reset_result.info
        telemetry = reset_result.telemetry
        self._episode.uses_custom_baseline = reset_result.uses_custom_baseline
        self.backend.set_controller_state(ControllerState())
        self._control_state.reset()
        self._mask_controller.set_lean_allowed_values(
            self._control_state.lean_action_mask_override()
        )
        self._mask_controller.set_spin_allowed_values(None)
        sync_dynamic_action_masks(
            mask_controller=self._mask_controller,
            control_state=self._control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self.config.boost_min_energy_fraction,
            mask_boost_when_active=self._action_config.mask_boost_when_active,
            mask_boost_when_airborne=self._action_config.mask_boost_when_airborne,
        )
        self._episode.last_telemetry = telemetry
        self._reward_tracker.reset(
            telemetry,
            episode_seed=self._reset_coordinator.reward_episode_seed(seed),
            course_id=None if selected_track is None else selected_track.course_id,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._step_assembler.reward_summary_config = self._reward_summary_config
        if isinstance(self._action_adapter, ResettableActionAdapter):
            self._action_adapter.reset()
        info["seed"] = seed
        set_curriculum_info(
            info,
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        set_episode_boost_pad_info(
            info,
            episode_boost_pad_entries=self._episode.boost_pad_entries,
        )
        info["episode_airborne_frames"] = self._episode.airborne_frames
        image_observation = self._observation_builder.render_image()
        observation = self._observation_builder.build_observation(
            image=image_observation,
            telemetry=telemetry,
            control_state=self._control_state,
        )
        self._observation_builder.set_info(
            info,
            image_shape=tuple(int(value) for value in image_observation.shape),
        )
        self._episode.last_info = dict(info)
        self._reset_coordinator.advance_reset_count()
        return observation, info

    def begin_live_race(
        self,
        *,
        seed: int | None = None,
        course_id: str | None = None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        """Initialize policy stepping from the emulator's current live race state.

        Career Mode reaches races through menu navigation instead of a materialized
        reset baseline. This method prepares only the policy/reward bookkeeping
        around that already-running emulator state.
        """

        self._episode.begin_reset(active_track=None)
        self._episode.uses_custom_baseline = False
        self.backend.set_controller_state(ControllerState())
        self._control_state.reset()
        self._mask_controller.set_lean_allowed_values(
            self._control_state.lean_action_mask_override()
        )
        self._mask_controller.set_spin_allowed_values(None)
        telemetry = self.backend.try_read_telemetry()
        sync_dynamic_action_masks(
            mask_controller=self._mask_controller,
            control_state=self._control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self.config.boost_min_energy_fraction,
            mask_boost_when_active=self._action_config.mask_boost_when_active,
            mask_boost_when_airborne=self._action_config.mask_boost_when_airborne,
        )
        self._episode.last_telemetry = telemetry
        self._reward_tracker.reset(
            telemetry,
            episode_seed=seed,
            course_id=course_id,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._step_assembler.reward_summary_config = self._reward_summary_config
        if isinstance(self._action_adapter, ResettableActionAdapter):
            self._action_adapter.reset()

        info = backend_step_info(self.backend)
        info["seed"] = seed
        set_curriculum_info(
            info,
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        set_episode_boost_pad_info(
            info,
            episode_boost_pad_entries=self._episode.boost_pad_entries,
        )
        info["episode_airborne_frames"] = self._episode.airborne_frames
        image_observation = self._observation_builder.render_image()
        observation = self._observation_builder.build_observation(
            image=image_observation,
            telemetry=telemetry,
            control_state=self._control_state,
        )
        self._observation_builder.set_info(
            info,
            image_shape=tuple(int(value) for value in image_observation.shape),
        )
        self._episode.last_info = dict(info)
        return observation, info

    def begin_policy_drive(
        self,
        *,
        seed: int | None = None,
        course_id: str | None = None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        """Initialize policy driving from the emulator's current race state."""

        observation, info = self.begin_live_race(seed=seed, course_id=course_id)
        return observation, policy_drive_info(info)

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_decoded_action(
            self._action_adapter.decode_request(action),
            action_drive_axis=action_drive_axis(
                action,
                self._action_space,
                drive_axis_index=self._action_config.continuous_drive_axis_index(),
            ),
        )

    def step_watch(self, action: ActionValue) -> WatchEnvStep:
        """Step a policy action while collecting watch-only intermediate frames."""

        return self._step_decoded_action_watch(
            self._action_adapter.decode_request(action),
            action_drive_axis=action_drive_axis(
                action,
                self._action_space,
                drive_axis_index=self._action_config.continuous_drive_axis_index(),
            ),
        )

    def step_policy_drive(self, action: ActionValue) -> PolicyDriveStep:
        """Step a live-race policy action without Gym lifecycle semantics."""

        return self._step_decoded_action_policy_drive(
            self._action_adapter.decode_request(action),
            action_drive_axis=action_drive_axis(
                action,
                self._action_space,
                drive_axis_index=self._action_config.continuous_drive_axis_index(),
            ),
        )

    def action_to_control_state(self, action: ActionValue) -> RaceControlState:
        """Decode one policy action into the held semantic control state."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: RaceControlState,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_control_state(control_state, action_drive_axis=None)

    def step_control_watch(self, control_state: RaceControlState) -> WatchEnvStep:
        """Step manual controls while collecting watch-only intermediate frames."""

        return self._step_control_state_watch(control_state, action_drive_axis=None)

    def step_control_policy_drive(
        self,
        control_state: RaceControlState,
    ) -> PolicyDriveStep:
        """Step live-race manual controls without Gym lifecycle semantics."""

        return self._step_control_state_policy_drive(
            control_state,
            action_drive_axis=None,
        )

    def _step_control_state(
        self,
        control_state: RaceControlState,
        *,
        action_drive_axis: float | None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request="none",
        )

    def _step_decoded_action(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        requested_control_state = decoded_action.control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        spin_request = self._apply_spin_semantics(decoded_action.spin_request)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
        )

    def _step_control_state_watch(
        self,
        control_state: RaceControlState,
        *,
        action_drive_axis: float | None,
    ) -> WatchEnvStep:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step_result(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request="none",
            capture_display_frames=True,
        )

    def _step_control_state_policy_drive(
        self,
        control_state: RaceControlState,
        *,
        action_drive_axis: float | None,
    ) -> PolicyDriveStep:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step_result(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request="none",
            capture_display_frames=True,
            episode_done_on_truncation=False,
        ).policy_drive_result()

    def _step_decoded_action_watch(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
    ) -> WatchEnvStep:
        requested_control_state = decoded_action.control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        spin_request = self._apply_spin_semantics(decoded_action.spin_request)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step_result(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
            capture_display_frames=True,
        )

    def _step_decoded_action_policy_drive(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
    ) -> PolicyDriveStep:
        requested_control_state = decoded_action.control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        spin_request = self._apply_spin_semantics(decoded_action.spin_request)
        self._episode.held_control_state = applied_control_state
        return self._run_env_step_result(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
            capture_display_frames=True,
            episode_done_on_truncation=False,
        ).policy_drive_result()

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        requested_control_state = (
            self._episode.held_control_state if control_state is None else control_state
        )
        self._episode.held_control_state = self._apply_control_semantics(requested_control_state)
        return self._run_env_step(
            self._episode.held_control_state,
            action_repeat=1,
            requested_control_state=requested_control_state,
            action_drive_axis=None,
            spin_request="none",
        )

    def render(self) -> RgbFrame:
        return self.backend.render_display(
            **self.config.observation.native_resolution_kwargs(renderer=self._renderer)
        )

    def close(self) -> None:
        self.backend.close()

    def _run_env_step(
        self,
        control_state: RaceControlState,
        *,
        action_repeat: int,
        requested_control_state: RaceControlState | None = None,
        action_drive_axis: float | None = None,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._run_env_step_result(
            control_state,
            action_repeat=action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
            capture_display_frames=False,
        ).gym_result()

    def _run_env_step_result(
        self,
        control_state: RaceControlState,
        *,
        action_repeat: int,
        requested_control_state: RaceControlState | None,
        action_drive_axis: float | None,
        spin_request: SpinRequest,
        capture_display_frames: bool,
        episode_done_on_truncation: bool = True,
    ) -> WatchEnvStep:
        assembly = self._step_assembler.run(
            EnvStepRequest(
                control_state=control_state,
                action_repeat=action_repeat,
                requested_control_state=requested_control_state,
                action_drive_axis=action_drive_axis,
                spin_request=spin_request,
                capture_display_frames=capture_display_frames,
                active_track=self._episode.active_track,
                episode_return=self._episode.return_value,
                episode_boost_pad_entries=self._episode.boost_pad_entries,
                episode_airborne_frames=self._episode.airborne_frames,
                curriculum_stage_index=self.curriculum_stage_index,
                curriculum_stage_name=self.curriculum_stage_name,
            )
        )
        self._episode.record_step(
            telemetry=assembly.telemetry,
            requested_control_state=assembly.requested_control_state,
            gas_level=assembly.gas_level,
            return_value=assembly.episode_return,
            boost_pad_entries=assembly.episode_boost_pad_entries,
            airborne_frames=assembly.episode_airborne_frames,
            done=assembly.step.terminated
            or (episode_done_on_truncation and assembly.step.truncated),
            info=assembly.step.info,
        )
        return assembly.step

    def _apply_control_semantics(self, control_state: RaceControlState) -> RaceControlState:
        """Apply telemetry gates and configured lean semantics."""

        gated_control_state = apply_dynamic_control_gates(
            control_state,
            mask_controller=self._mask_controller,
            mask_air_brake_on_ground=self._action_config.mask_air_brake_on_ground,
            continuous_air_brake_mode=self._action_config.continuous_air_brake_mode,
            last_telemetry=self._episode.last_telemetry,
        )
        return self._control_state.apply_lean_semantics(gated_control_state)

    def _apply_spin_semantics(self, spin_request: SpinRequest) -> SpinRequest:
        """Suppress spin requests currently masked by runtime gates."""

        if spin_request == "none":
            return "none"
        spin_index = 1 if spin_request == "left" else 2
        if action_branch_value_allowed(
            self._mask_controller.action_mask_branches(),
            "spin",
            spin_index,
            missing_allowed=True,
        ):
            return spin_request
        return "none"


_RENDERERS_BY_NAME: dict[str, RendererName] = {name: name for name in KNOWN_RENDERERS}


def _backend_renderer(backend: EmulatorBackend) -> RendererName:
    """Return the backend renderer name used for renderer-sized observations."""

    renderer = getattr(backend, "renderer", DEFAULT_RENDERER)
    return _RENDERERS_BY_NAME.get(str(renderer), DEFAULT_RENDERER)
