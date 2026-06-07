# src/rl_fzerox/core/envs/gym_runtime/runtime.py
from __future__ import annotations

from collections.abc import Sequence

from gymnasium import spaces

from fzerox_emulator import EmulatorBackend, RaceControlState, SpinRequest
from fzerox_emulator.arrays import ActionMask, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import (
    ActionValue,
    DecodedAction,
    DiscreteActionDimension,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.engine.components import build_engine_runtime_components
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    ActionMaskSnapshot,
    apply_control_semantics,
    apply_spin_semantics,
)
from rl_fzerox.core.envs.engine.reset import EngineResetCoordinator
from rl_fzerox.core.envs.engine.stepping import (
    EnvStepRequest,
    WatchEnvStep,
)
from rl_fzerox.core.envs.gym_runtime.reset import reset_gym_episode
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_vector_or_zeros,
)
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)


class FZeroXEnvRuntime:
    """Training/watch runtime around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply Gym termination,
    reward shaping, policy observations, and watch-facing info.
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
        self._components = build_engine_runtime_components(
            backend=backend,
            config=config,
            reward_config=reward_config,
            curriculum_config=curriculum_config,
        )
        components = self._components
        self._renderer: RendererName = components.renderer
        self._action_config = components.action_config
        self._action_adapter = components.action_adapter
        self._observation_builder = components.observation_builder
        self._reward_tracker = components.reward_tracker
        self._action_space = self._action_adapter.action_space
        self._observation_space = self._observation_builder.space
        self._mask_controller = components.mask_controller
        self._reset_coordinator = EngineResetCoordinator(
            backend=backend,
            config=config,
            curriculum_config=curriculum_config,
            env_index=env_index,
        )
        self._reset_coordinator.set_curriculum_stage(self._mask_controller.stage_index)
        self._control_state = components.control_state
        self._step_assembler = components.step_assembler
        self._episode = components.episode

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

        return reset_gym_episode(
            backend=self.backend,
            config=self.config,
            components=self._components,
            reset_coordinator=self._reset_coordinator,
            seed=seed,
        )

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
            done=assembly.step.terminated or assembly.step.truncated,
            info=assembly.step.info,
        )
        return assembly.step

    def _apply_control_semantics(self, control_state: RaceControlState) -> RaceControlState:
        """Apply telemetry gates and configured lean semantics."""

        return apply_control_semantics(
            control_state,
            mask_controller=self._mask_controller,
            action_config=self._action_config,
            control_state_tracker=self._control_state,
            last_telemetry=self._episode.last_telemetry,
        )

    def _apply_spin_semantics(self, spin_request: SpinRequest) -> SpinRequest:
        """Suppress spin requests currently masked by runtime gates."""

        return apply_spin_semantics(
            spin_request,
            mask_controller=self._mask_controller,
        )
