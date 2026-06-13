# src/rl_fzerox/core/envs/policy_drive/runtime.py
"""Policy-drive runtime around shared emulator stepping components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator import ControllerState, RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import (
    ActionValue,
    DecodedAction,
    ResettableActionAdapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.engine.components import build_engine_runtime_components
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    ActionMaskSnapshot,
    apply_control_semantics,
    apply_spin_semantics,
    sync_dynamic_action_masks,
)
from rl_fzerox.core.envs.engine.controls.episode_dropout import sample_episode_lean_mask
from rl_fzerox.core.envs.engine.info import backend_step_info, set_curriculum_info, telemetry_info
from rl_fzerox.core.envs.engine.reset import EngineResetSeeds
from rl_fzerox.core.envs.engine.stepping import (
    EnvStepRequest,
    set_episode_boost_pad_info,
)
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.policy_drive.frame import (
    PolicyDriveFrame,
    PolicyDriveStep,
    policy_drive_step,
)

if TYPE_CHECKING:
    from fzerox_emulator import EmulatorBackend
    from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


class PolicyDriveRuntime:
    """Policy-race adapter with no Gym episode lifecycle contract."""

    def __init__(
        self,
        *,
        emulator: EmulatorBackend,
        train_config: TrainAppConfig,
    ) -> None:
        self.train_config = train_config
        self._backend = emulator
        self._config = train_config.env
        components = build_engine_runtime_components(
            backend=emulator,
            config=self._config,
            reward_config=train_config.reward,
            curriculum_config=train_config.curriculum,
        )
        self._renderer = components.renderer
        self._action_config = components.action_config
        self._action_adapter = components.action_adapter
        self._observation_builder = components.observation_builder
        self._reward_tracker = components.reward_tracker
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._mask_controller = components.mask_controller
        self._control_state = components.control_state
        self._step_assembler = components.step_assembler
        self._episode = components.episode
        self._reset_seeds = EngineResetSeeds()

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
        self._reset_seeds.remember_reset_seed(seed)
        self._episode.begin_reset(active_track=None)
        self._episode.uses_custom_baseline = False
        self._backend.set_controller_state(ControllerState())
        self._control_state.reset()
        self._episode.lean_episode_masked = sample_episode_lean_mask(
            probability=self._action_config.lean_episode_mask_probability,
            seed=self._reset_seeds.action_episode_mask_seed(seed),
        )
        self._mask_controller.set_lean_episode_masked(self._episode.lean_episode_masked)
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
        self._reward_tracker.reset(
            telemetry,
            episode_seed=self._reset_seeds.reward_episode_seed(seed),
            course_id=course_id,
        )
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
        info["episode_step"] = self._episode.frame_count
        info["episode_return"] = self._episode.return_value
        info["episode_airborne_frames"] = self._episode.airborne_frames
        info["lean_episode_masked"] = self._episode.lean_episode_masked
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
        self._reset_seeds.advance_reset_count()
        return observation, info

    def step_policy(
        self,
        action: ActionValue,
        *,
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return _policy_drive_frame(
            self._step_decoded_action(
                self._action_adapter.decode_request(action),
                action_drive_axis=action_drive_axis(
                    action,
                    self._action_adapter.action_space,
                    drive_axis_index=self._action_config.continuous_drive_axis_index(),
                ),
                capture_audio=capture_audio,
            )
        )

    def step_manual(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return _policy_drive_frame(
            self._step_control_state(
                control_state,
                action_drive_axis=None,
                spin_request=spin_request,
                capture_audio=capture_audio,
            )
        )

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
        spin_request: SpinRequest = "none",
        capture_audio: bool = False,
    ) -> PolicyDriveStep:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_control_state = applied_control_state
        return self._run_step(
            applied_control_state,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            spin_request=spin_request,
            capture_audio=capture_audio,
        )

    def _step_decoded_action(
        self,
        decoded_action: DecodedAction,
        *,
        action_drive_axis: float | None,
        capture_audio: bool = False,
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
            capture_audio=capture_audio,
        )

    def _run_step(
        self,
        control_state: RaceControlState,
        *,
        requested_control_state: RaceControlState,
        action_drive_axis: float | None,
        spin_request: SpinRequest,
        capture_audio: bool,
    ) -> PolicyDriveStep:
        assembly = self._step_assembler.run_live_race(
            EnvStepRequest(
                control_state=control_state,
                action_repeat=self._config.action_repeat,
                requested_control_state=requested_control_state,
                action_drive_axis=action_drive_axis,
                spin_request=spin_request,
                capture_display_frames=True,
                capture_audio=capture_audio,
                active_track=None,
                episode_frame_count=self._episode.frame_count,
                episode_stalled_steps=self._episode.stalled_steps,
                episode_progress_frontier_stalled_frames=(
                    self._episode.progress_frontier_stalled_frames
                ),
                episode_progress_frontier_distance=self._episode.progress_frontier_distance,
                episode_progress_frontier_initialized=(self._episode.progress_frontier_initialized),
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
            frame_count=assembly.episode_frame_count,
            stalled_steps=assembly.episode_stalled_steps,
            progress_frontier_stalled_frames=(assembly.episode_progress_frontier_stalled_frames),
            progress_frontier_distance=assembly.episode_progress_frontier_distance,
            progress_frontier_initialized=(assembly.episode_progress_frontier_initialized),
            return_value=assembly.episode_return,
            boost_pad_entries=assembly.episode_boost_pad_entries,
            airborne_frames=assembly.episode_airborne_frames,
            done=assembly.step.terminated,
            info=assembly.step.info,
        )
        return policy_drive_step(assembly.step)

    def _apply_control_semantics(self, control_state: RaceControlState) -> RaceControlState:
        return apply_control_semantics(
            control_state,
            mask_controller=self._mask_controller,
            action_config=self._action_config,
            control_state_tracker=self._control_state,
            last_telemetry=self._episode.last_telemetry,
        )

    def _apply_spin_semantics(self, spin_request: SpinRequest) -> SpinRequest:
        return apply_spin_semantics(
            spin_request,
            mask_controller=self._mask_controller,
        )


def _policy_drive_frame(step: PolicyDriveStep) -> PolicyDriveFrame:
    return PolicyDriveFrame(
        observation=step.observation,
        reward=step.reward,
        info=dict(step.info),
        display_frames=step.display_frames,
        display_controller_masks=step.display_controller_masks,
        audio_samples=step.audio_samples,
        audio_frame_counts=step.audio_frame_counts,
    )
