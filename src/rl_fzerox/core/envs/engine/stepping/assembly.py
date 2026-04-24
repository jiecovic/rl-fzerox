# src/rl_fzerox/core/envs/engine/stepping/assembly.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import BackendStepResult, ControllerState, EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.config.schema_models.actions import ActionRuntimeConfig
from rl_fzerox.core.domain.lean import LEAN_MODE_TIMER_ASSIST
from rl_fzerox.core.envs.actions import (
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.core.envs.actions.continuous_controls import requested_gas_level
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.rewards import RewardActionContext, RewardSummaryConfig, RewardTracker

from ..controls import ActionMaskController, ControlStateTracker, sync_dynamic_action_masks
from ..info import backend_step_info, set_curriculum_info, telemetry_info
from ..observation import EngineObservationBuilder
from ..reset import SelectedTrack
from .result import WatchEnvStep


@dataclass(frozen=True, slots=True)
class EnvStepRequest:
    """Inputs for one outer env step after controller semantics were applied."""

    control_state: ControllerState
    action_repeat: int
    requested_control_state: ControllerState | None
    action_drive_axis: float | None
    capture_display_frames: bool
    active_track: SelectedTrack | None
    episode_return: float
    episode_boost_pad_entries: int
    curriculum_stage_index: int | None
    curriculum_stage_name: str | None


@dataclass(frozen=True, slots=True)
class EnvStepAssembly:
    """Step result plus engine fields that runtime must persist."""

    step: WatchEnvStep
    telemetry: FZeroXTelemetry | None
    requested_control_state: ControllerState
    gas_level: float
    episode_return: float
    episode_boost_pad_entries: int


@dataclass(slots=True)
class EngineStepAssembler:
    """Assemble one native repeated-step result into Gym/watch env outputs."""

    backend: EmulatorBackend
    config: EnvConfig
    action_config: ActionRuntimeConfig
    reward_summary_config: RewardSummaryConfig
    reward_tracker: RewardTracker
    observation_builder: EngineObservationBuilder
    mask_controller: ActionMaskController
    control_state: ControlStateTracker

    def run(self, request: EnvStepRequest) -> EnvStepAssembly:
        requested_control_state = request.requested_control_state or request.control_state
        applied_control_state = request.control_state
        step_result = self._native_step(applied_control_state, request)

        info = backend_step_info(self.backend)
        if request.active_track is not None:
            info.update(request.active_track.info())

        telemetry = step_result.telemetry
        gas_level = requested_gas_level(
            control_state=requested_control_state,
            drive_axis=request.action_drive_axis,
            continuous_drive_deadzone=float(self.action_config.continuous_drive_deadzone),
            continuous_drive_full_threshold=float(
                self.action_config.continuous_drive_full_threshold
            ),
            continuous_drive_min_thrust=float(self.action_config.continuous_drive_min_thrust),
        )

        boost_used = bool(applied_control_state.joypad_mask & BOOST_MASK)
        lean_used = bool(applied_control_state.joypad_mask & (LEAN_LEFT_MASK | LEAN_RIGHT_MASK))
        air_brake_used = bool(applied_control_state.joypad_mask & AIR_BRAKE_MASK)
        reward_step = self.reward_tracker.step_summary(
            step_result.summary,
            step_result.status,
            telemetry,
            RewardActionContext(
                air_brake_requested=air_brake_used,
                boost_requested=boost_used,
                lean_requested=lean_used,
                gas_level=gas_level,
                steer_level=max(
                    -1.0,
                    min(1.0, float(requested_control_state.left_stick_x)),
                ),
                drive_axis=request.action_drive_axis,
            ),
        )
        reward = reward_step.reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = step_result.status.terminated
        truncated = step_result.status.truncated

        episode_boost_pad_entries = request.episode_boost_pad_entries
        boost_pad_entered = bool(step_result.summary.entered_dash_pad_boost)
        if boost_pad_entered:
            episode_boost_pad_entries += 1

        info["step_reward"] = reward
        info["frames_run"] = int(step_result.summary.frames_run)
        info["repeat_index"] = max(step_result.summary.frames_run - 1, 0)
        info["energy_loss_total"] = float(step_result.summary.energy_loss_total)
        info["damage_taken_frames"] = int(step_result.summary.damage_taken_frames)
        info["airborne_frames"] = int(step_result.summary.airborne_frames)
        info["collision_recoil_entered"] = bool(step_result.summary.entered_collision_recoil)
        info["boost_pad_entered"] = boost_pad_entered
        info["boost_used"] = boost_used
        info["lean_used"] = lean_used
        if reward_breakdown:
            info["reward_breakdown"] = reward_breakdown
        info["episode_step"] = step_result.status.step_count
        info["stuck_truncation_enabled"] = self.config.stuck_truncation_enabled
        info["stalled_steps"] = step_result.status.stalled_steps
        info["reverse_timer"] = step_result.status.reverse_timer
        info["progress_frontier_stalled_frames"] = (
            step_result.status.progress_frontier_stalled_frames
        )
        info["termination_reason"] = step_result.status.termination_reason
        info["truncation_reason"] = step_result.status.truncation_reason
        set_curriculum_info(
            info,
            stage_index=request.curriculum_stage_index,
            stage_name=request.curriculum_stage_name,
        )
        if telemetry is not None:
            # Keep env info pickle-safe for SubprocVecEnv workers.
            info.update(telemetry_info(telemetry))
        info.update(self.reward_tracker.info(telemetry))
        set_episode_boost_pad_info(info, episode_boost_pad_entries=episode_boost_pad_entries)

        self.control_state.record_step(
            control_state=applied_control_state,
            frames_run=step_result.summary.frames_run,
            gas_level=gas_level,
        )
        self.mask_controller.set_lean_allowed_values(
            self.control_state.lean_action_mask_override(),
        )
        sync_dynamic_action_masks(
            mask_controller=self.mask_controller,
            control_state=self.control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self.config.boost_min_energy_fraction,
        )

        image_observation = step_result.observation
        observation = self.observation_builder.build_observation(
            image=image_observation,
            telemetry=telemetry,
            control_state=self.control_state,
        )
        episode_return = request.episode_return + reward
        info["episode_return"] = episode_return
        ensure_monitor_info_keys(info)
        self.observation_builder.set_info(
            info,
            image_shape=tuple(int(value) for value in image_observation.shape),
        )

        return EnvStepAssembly(
            step=WatchEnvStep(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                display_frames=step_result.display_frames,
            ),
            telemetry=telemetry,
            requested_control_state=requested_control_state,
            gas_level=gas_level,
            episode_return=episode_return,
            episode_boost_pad_entries=episode_boost_pad_entries,
        )

    def _native_step(
        self,
        control_state: ControllerState,
        request: EnvStepRequest,
    ) -> BackendStepResult:
        stuck_step_limit = (
            self.config.stuck_step_limit
            if self.config.stuck_truncation_enabled
            else self.config.max_episode_steps + 1
        )
        wrong_way_timer_limit = (
            self.config.wrong_way_timer_limit if self.config.wrong_way_truncation_enabled else None
        )
        lean_timer_assist = self.action_config.lean_mode == LEAN_MODE_TIMER_ASSIST
        if request.capture_display_frames:
            return self.backend.step_repeat_watch_raw(
                control_state,
                action_repeat=request.action_repeat,
                preset=self.config.observation.preset,
                frame_stack=self.config.observation.frame_stack,
                stack_mode=self.config.observation.stack_mode,
                minimap_layer=self.config.observation.minimap_layer,
                resize_filter=self.config.observation.resize_filter,
                minimap_resize_filter=self.config.observation.minimap_resize_filter,
                stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
                energy_loss_epsilon=self.reward_summary_config.energy_loss_epsilon,
                max_episode_steps=self.config.max_episode_steps,
                stuck_step_limit=stuck_step_limit,
                wrong_way_timer_limit=wrong_way_timer_limit,
                progress_frontier_stall_limit_frames=(
                    self.config.progress_frontier_stall_limit_frames
                ),
                progress_frontier_epsilon=float(self.config.progress_frontier_epsilon),
                terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
                lean_timer_assist=lean_timer_assist,
            )
        return self.backend.step_repeat_raw(
            control_state,
            action_repeat=request.action_repeat,
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
            stack_mode=self.config.observation.stack_mode,
            minimap_layer=self.config.observation.minimap_layer,
            resize_filter=self.config.observation.resize_filter,
            minimap_resize_filter=self.config.observation.minimap_resize_filter,
            stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
            energy_loss_epsilon=self.reward_summary_config.energy_loss_epsilon,
            max_episode_steps=self.config.max_episode_steps,
            stuck_step_limit=stuck_step_limit,
            wrong_way_timer_limit=(
                self.config.wrong_way_timer_limit
                if self.config.wrong_way_truncation_enabled
                else None
            ),
            progress_frontier_stall_limit_frames=(self.config.progress_frontier_stall_limit_frames),
            progress_frontier_epsilon=float(self.config.progress_frontier_epsilon),
            terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
        )


def set_episode_boost_pad_info(
    info: dict[str, object],
    *,
    episode_boost_pad_entries: int,
) -> None:
    """Attach episode-local boost-pad counts for Monitor/TensorBoard."""

    info["boost_pad_entries"] = episode_boost_pad_entries
    laps_completed = info.get("race_laps_completed")
    if not isinstance(laps_completed, int | float) or laps_completed <= 0:
        info["boost_pad_entries_per_lap"] = None
        return
    info["boost_pad_entries_per_lap"] = episode_boost_pad_entries / float(laps_completed)
