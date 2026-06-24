# src/rl_fzerox/core/envs/gym_runtime/stepping.py
from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import (
    ActionValue,
    DecodedAction,
    DiscreteActionDimension,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.engine.components import EngineRuntimeComponents
from rl_fzerox.core.envs.engine.controls import (
    apply_control_semantics,
    apply_spin_semantics,
)
from rl_fzerox.core.envs.engine.stepping import EnvStepRequest, WatchEnvStep
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.runtime_spec.schema import EnvConfig


@dataclass(frozen=True)
class _GymStepInput:
    control_state: RaceControlState
    action_repeat: int
    requested_control_state: RaceControlState
    action_drive_axis: float | None
    spin_request: SpinRequest


class GymStepRuntime:
    """Training/watch step contract around shared low-level step assembly."""

    def __init__(
        self,
        *,
        config: EnvConfig,
        components: EngineRuntimeComponents,
    ) -> None:
        self._config = config
        self._components = components

    @property
    def action_space(self) -> spaces.Space:
        return self._components.action_adapter.action_space

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        return self._components.action_adapter.action_dimensions

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._run_step(
            self._policy_step_input(action),
            capture_display_frames=False,
        ).gym_result()

    def step_watch(self, action: ActionValue) -> WatchEnvStep:
        """Step a policy action while collecting watch-only intermediate frames."""

        return self._run_step(
            self._policy_step_input(action),
            capture_display_frames=True,
        )

    def action_to_control_state(self, action: ActionValue) -> RaceControlState:
        """Decode one policy action into the held semantic control state."""

        return self._components.action_adapter.decode(action)

    def step_control(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._run_step(
            self._control_step_input(
                control_state,
                action_repeat=self._config.action_repeat,
                action_drive_axis=None,
                spin_request=spin_request,
            ),
            capture_display_frames=False,
        ).gym_result()

    def step_control_watch(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ) -> WatchEnvStep:
        """Step manual controls while collecting watch-only intermediate frames."""

        return self._run_step(
            self._control_step_input(
                control_state,
                action_repeat=self._config.action_repeat,
                action_drive_axis=None,
                spin_request=spin_request,
            ),
            capture_display_frames=True,
        )

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
        *,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        requested_control_state = (
            self._components.episode.held_control_state
            if control_state is None
            else control_state
        )
        return self._run_step(
            self._control_step_input(
                requested_control_state,
                action_repeat=1,
                action_drive_axis=None,
                spin_request=spin_request,
            ),
            capture_display_frames=False,
        ).gym_result()

    def _policy_step_input(self, action: ActionValue) -> _GymStepInput:
        return self._decoded_step_input(
            self._components.action_adapter.decode_request(action),
            action_drive_axis=action_drive_axis(
                action,
                self._components.action_adapter.action_space,
                drive_axis_index=self._components.action_config.continuous_drive_axis_index(),
            ),
        )

    def _control_step_input(
        self,
        control_state: RaceControlState,
        *,
        action_repeat: int,
        action_drive_axis: float | None,
        spin_request: SpinRequest,
    ) -> _GymStepInput:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        return self._prepared_step_input(
            applied_control_state,
            action_repeat=action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
        )

    def _decoded_step_input(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
    ) -> _GymStepInput:
        requested_control_state = decoded_action.control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        spin_request = self._apply_spin_semantics(decoded_action.spin_request)
        return self._prepared_step_input(
            applied_control_state,
            action_repeat=self._config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
        )

    def _prepared_step_input(
        self,
        applied_control_state: RaceControlState,
        *,
        action_repeat: int,
        requested_control_state: RaceControlState,
        action_drive_axis: float | None,
        spin_request: SpinRequest,
    ) -> _GymStepInput:
        self._components.episode.held_control_state = applied_control_state
        return _GymStepInput(
            control_state=applied_control_state,
            action_repeat=action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
        )

    def _run_step(
        self,
        step_input: _GymStepInput,
        *,
        capture_display_frames: bool,
    ) -> WatchEnvStep:
        components = self._components
        assembly = components.step_assembler.run(
            EnvStepRequest(
                control_state=step_input.control_state,
                action_repeat=step_input.action_repeat,
                requested_control_state=step_input.requested_control_state,
                action_drive_axis=step_input.action_drive_axis,
                spin_request=step_input.spin_request,
                capture_display_frames=capture_display_frames,
                capture_audio=False,
                active_track=components.episode.active_track,
                episode_frame_count=components.episode.frame_count,
                episode_stalled_steps=components.episode.stalled_steps,
                episode_progress_frontier_stalled_frames=(
                    components.episode.progress_frontier_stalled_frames
                ),
                episode_progress_frontier_distance=(components.episode.progress_frontier_distance),
                episode_progress_frontier_initialized=(
                    components.episode.progress_frontier_initialized
                ),
                episode_return=components.episode.return_value,
                episode_boost_pad_entries=components.episode.boost_pad_entries,
                episode_airborne_frames=components.episode.airborne_frames,
            )
        )
        components.episode.record_step(
            telemetry=assembly.telemetry,
            requested_control_state=assembly.requested_control_state,
            gas_level=assembly.gas_level,
            frame_count=assembly.episode_frame_count,
            stalled_steps=assembly.episode_stalled_steps,
            progress_frontier_stalled_frames=(assembly.episode_progress_frontier_stalled_frames),
            progress_frontier_distance=assembly.episode_progress_frontier_distance,
            progress_frontier_initialized=(assembly.episode_progress_frontier_initialized),
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
            mask_controller=self._components.mask_controller,
            action_config=self._components.action_config,
            control_state_tracker=self._components.control_state,
            last_telemetry=self._components.episode.last_telemetry,
        )

    def _apply_spin_semantics(self, spin_request: SpinRequest) -> SpinRequest:
        """Suppress spin requests currently masked by runtime gates."""

        return apply_spin_semantics(
            spin_request,
            mask_controller=self._components.mask_controller,
        )
