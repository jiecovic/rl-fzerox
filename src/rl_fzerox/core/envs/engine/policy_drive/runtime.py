# src/rl_fzerox/core/envs/engine/policy_drive/runtime.py
"""Policy-drive runtime around shared emulator stepping components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator import ControllerState, RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import (
    ActionValue,
    DecodedAction,
    ResettableActionAdapter,
    build_action_adapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    ActionMaskController,
    ActionMaskSnapshot,
    ControlStateTracker,
    action_branch_value_allowed,
    apply_dynamic_control_gates,
    sync_dynamic_action_masks,
)
from rl_fzerox.core.envs.engine.episode import EngineEpisodeState
from rl_fzerox.core.envs.engine.info import backend_step_info, set_curriculum_info, telemetry_info
from rl_fzerox.core.envs.engine.observation import EngineObservationBuilder
from rl_fzerox.core.envs.engine.policy_drive.frame import PolicyDriveFrame
from rl_fzerox.core.envs.engine.rendering import backend_renderer
from rl_fzerox.core.envs.engine.stepping import (
    EngineStepAssembler,
    EnvStepRequest,
    PolicyDriveStep,
    set_episode_boost_pad_info,
)
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.rewards import build_reward_tracker

if TYPE_CHECKING:
    from fzerox_emulator import Emulator
    from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


class PolicyDriveRuntime:
    """Policy-race adapter with no Gym episode lifecycle contract."""

    def __init__(
        self,
        *,
        emulator: Emulator,
        train_config: TrainAppConfig,
    ) -> None:
        self.train_config = train_config
        self._backend = emulator
        self._config = train_config.env
        self._action_config = self._config.action.runtime()
        self._renderer = backend_renderer(emulator)
        self._action_adapter = build_action_adapter(self._action_config)
        self._observation_builder = EngineObservationBuilder.from_engine_config(
            backend=emulator,
            config=self._config,
            renderer=self._renderer,
        )
        self._reward_tracker = build_reward_tracker(
            config=train_config.reward,
            max_episode_steps=self._config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._mask_controller = ActionMaskController.from_config(
            adapter=self._action_adapter,
            base_overrides=self._action_config.mask_overrides,
            curriculum_config=train_config.curriculum,
            boost_unmask_max_speed_kph=self._action_config.boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=self._action_config.lean_unmask_min_speed_kph,
            mask_air_brake_on_ground=self._action_config.mask_air_brake_on_ground,
            pitch_neutral_index=self._action_config.pitch_buckets // 2,
        )
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
            backend=emulator,
            config=self._config,
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
    def last_requested_control_state(self) -> RaceControlState:
        return self._episode.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._episode.last_gas_level

    def begin(
        self,
        *,
        seed: int | None,
        course_id: str | None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        self._episode.begin_reset(active_track=None)
        self._episode.uses_custom_baseline = False
        self._backend.set_controller_state(ControllerState())
        self._control_state.reset()
        self._mask_controller.set_lean_allowed_values(
            self._control_state.lean_action_mask_override()
        )
        self._mask_controller.set_spin_allowed_values(None)
        telemetry = self._backend.try_read_telemetry()
        sync_dynamic_action_masks(
            mask_controller=self._mask_controller,
            control_state=self._control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self._config.boost_min_energy_fraction,
            mask_boost_when_active=self._action_config.mask_boost_when_active,
            mask_boost_when_airborne=self._action_config.mask_boost_when_airborne,
        )
        self._episode.last_telemetry = telemetry
        self._reward_tracker.reset(telemetry, episode_seed=seed, course_id=course_id)
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._step_assembler.reward_summary_config = self._reward_summary_config
        if isinstance(self._action_adapter, ResettableActionAdapter):
            self._action_adapter.reset()

        info = backend_step_info(self._backend)
        info["seed"] = seed
        set_curriculum_info(
            info,
            stage_index=self._mask_controller.stage_index,
            stage_name=self._mask_controller.stage_name,
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

    def step_policy(self, action: ActionValue) -> PolicyDriveFrame:
        return _policy_drive_frame(
            self._step_decoded_action(
                self._action_adapter.decode_request(action),
                action_drive_axis=action_drive_axis(
                    action,
                    self._action_adapter.action_space,
                    drive_axis_index=self._action_config.continuous_drive_axis_index(),
                ),
            )
        )

    def step_manual(self, control_state: RaceControlState) -> PolicyDriveFrame:
        return _policy_drive_frame(self._step_control_state(control_state, action_drive_axis=None))

    def action_mask_branches(self) -> ActionMaskBranches:
        return self._mask_controller.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._mask_controller.action_mask_snapshot()

    def sync_curriculum_stage(self, stage_index: int | None) -> None:
        self._mask_controller.sync_checkpoint_stage(stage_index)

    def _step_control_state(
        self,
        control_state: RaceControlState,
        *,
        action_drive_axis: float | None,
    ) -> PolicyDriveStep:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_control_state = applied_control_state
        return self._run_step(
            applied_control_state,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request="none",
        )

    def _step_decoded_action(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
    ) -> PolicyDriveStep:
        requested_control_state = decoded_action.control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        spin_request = self._apply_spin_semantics(decoded_action.spin_request)
        self._episode.held_control_state = applied_control_state
        return self._run_step(
            applied_control_state,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
        )

    def _run_step(
        self,
        control_state: RaceControlState,
        *,
        requested_control_state: RaceControlState,
        action_drive_axis: float | None,
        spin_request: SpinRequest,
    ) -> PolicyDriveStep:
        assembly = self._step_assembler.run(
            EnvStepRequest(
                control_state=control_state,
                action_repeat=self._config.action_repeat,
                requested_control_state=requested_control_state,
                action_drive_axis=action_drive_axis,
                spin_request=spin_request,
                capture_display_frames=True,
                active_track=None,
                episode_return=self._episode.return_value,
                episode_boost_pad_entries=self._episode.boost_pad_entries,
                episode_airborne_frames=self._episode.airborne_frames,
                curriculum_stage_index=self._mask_controller.stage_index,
                curriculum_stage_name=self._mask_controller.stage_name,
            )
        )
        self._episode.record_step(
            telemetry=assembly.telemetry,
            requested_control_state=assembly.requested_control_state,
            gas_level=assembly.gas_level,
            return_value=assembly.episode_return,
            boost_pad_entries=assembly.episode_boost_pad_entries,
            airborne_frames=assembly.episode_airborne_frames,
            done=assembly.step.terminated,
            info=assembly.step.info,
        )
        return assembly.step.policy_drive_result()

    def _apply_control_semantics(self, control_state: RaceControlState) -> RaceControlState:
        gated_control_state = apply_dynamic_control_gates(
            control_state,
            mask_controller=self._mask_controller,
            mask_air_brake_on_ground=self._action_config.mask_air_brake_on_ground,
            continuous_air_brake_mode=self._action_config.continuous_air_brake_mode,
            last_telemetry=self._episode.last_telemetry,
        )
        return self._control_state.apply_lean_semantics(gated_control_state)

    def _apply_spin_semantics(self, spin_request: SpinRequest) -> SpinRequest:
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


def _policy_drive_frame(step: PolicyDriveStep) -> PolicyDriveFrame:
    return PolicyDriveFrame(
        observation=step.observation,
        reward=step.reward,
        info=dict(step.info),
        display_frames=step.display_frames,
        display_controller_masks=step.display_controller_masks,
    )
