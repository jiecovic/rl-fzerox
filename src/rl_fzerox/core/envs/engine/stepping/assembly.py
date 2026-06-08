# src/rl_fzerox/core/envs/engine/stepping/assembly.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import (
    BackendStepResult,
    EmulatorBackend,
    FZeroXTelemetry,
    RaceControlState,
    SpinRequest,
    StepStatus,
)
from fzerox_emulator.boundary import StepStatusDict
from rl_fzerox.core.envs.actions.continuous_controls import requested_gas_level
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.rewards import RewardActionContext, RewardSummaryConfig, RewardTracker
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema import EnvConfig
from rl_fzerox.core.runtime_spec.schema.actions import ActionRuntimeConfig

from ..controls import ActionMaskController, ControlStateTracker, sync_dynamic_action_masks
from ..info import backend_step_info, set_curriculum_info, telemetry_info
from ..observation import EngineObservationBuilder
from ..reset import SelectedTrack
from .result import WatchEnvStep


@dataclass(frozen=True, slots=True)
class EnvStepRequest:
    """Inputs for one outer env step after controller semantics were applied."""

    control_state: RaceControlState
    action_repeat: int
    requested_control_state: RaceControlState | None
    action_drive_axis: float | None
    spin_request: SpinRequest
    capture_display_frames: bool
    active_track: SelectedTrack | None
    episode_frame_count: int
    episode_stalled_steps: int
    episode_progress_frontier_stalled_frames: int
    episode_progress_frontier_distance: float
    episode_progress_frontier_initialized: bool
    episode_return: float
    episode_boost_pad_entries: int
    episode_airborne_frames: int
    curriculum_stage_index: int | None
    curriculum_stage_name: str | None


@dataclass(frozen=True, slots=True)
class EnvStepAssembly:
    """Step result plus engine fields that runtime must persist."""

    step: WatchEnvStep
    telemetry: FZeroXTelemetry | None
    requested_control_state: RaceControlState
    gas_level: float
    episode_frame_count: int
    episode_stalled_steps: int
    episode_progress_frontier_stalled_frames: int
    episode_progress_frontier_distance: float
    episode_progress_frontier_initialized: bool
    episode_return: float
    episode_boost_pad_entries: int
    episode_airborne_frames: int


@dataclass(frozen=True, slots=True)
class _EpisodeStepStatus:
    status: StepStatus
    progress_frontier_distance: float
    progress_frontier_initialized: bool


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
    renderer: RendererName

    def run(self, request: EnvStepRequest) -> EnvStepAssembly:
        return self._run(
            request,
            live_race_status=False,
        )

    def run_live_race(self, request: EnvStepRequest) -> EnvStepAssembly:
        """Assemble a live race step without Gym truncation semantics."""

        return self._run(
            request,
            live_race_status=True,
        )

    def _run(
        self,
        request: EnvStepRequest,
        *,
        live_race_status: bool,
    ) -> EnvStepAssembly:
        requested_control_state = request.requested_control_state or request.control_state
        applied_control_state = request.control_state
        step_result = self._native_step(applied_control_state, request)
        episode_status = (
            _live_race_episode_status(
                request=request,
                step_result=step_result,
                config=self.config,
            )
            if live_race_status
            else _native_episode_status(
                request=request,
                step_result=step_result,
                config=self.config,
            )
        )
        status = episode_status.status

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

        boost_used = applied_control_state.boost
        lean_used = applied_control_state.lean_left or applied_control_state.lean_right
        spin_requested = request.spin_request != "none"
        spin_started = bool(step_result.summary.spin_macro_started)
        spin_active_frames = int(step_result.summary.spin_macro_active_frames)
        air_brake_used = applied_control_state.air_brake
        reward_step = self.reward_tracker.step_summary(
            step_result.summary,
            status,
            telemetry,
            RewardActionContext(
                air_brake_requested=air_brake_used,
                boost_requested=boost_used,
                lean_requested=lean_used,
                spin_requested=spin_requested,
                gas_level=gas_level,
                steer_level=max(
                    -1.0,
                    min(1.0, float(requested_control_state.stick_x)),
                ),
                pitch_level=max(
                    -1.0,
                    min(1.0, float(requested_control_state.pitch)),
                ),
                pitch_deadzone=float(self.action_config.pitch_deadzone),
                drive_axis=request.action_drive_axis,
            ),
        )
        reward = reward_step.reward
        raw_reward = reward if reward_step.raw_reward is None else reward_step.raw_reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = status.terminated
        truncated = status.truncated

        episode_boost_pad_entries = request.episode_boost_pad_entries
        boost_pad_entered = bool(step_result.summary.entered_dash_surface)
        if boost_pad_entered:
            episode_boost_pad_entries += 1
        episode_airborne_frames = request.episode_airborne_frames + int(
            step_result.summary.airborne_frames
        )

        info["step_reward"] = reward
        info["step_reward_raw"] = raw_reward
        info["step_reward_clipped"] = raw_reward != reward
        info["step_reward_clip_delta"] = reward - raw_reward
        info["step_reward_clip_abs_excess"] = abs(reward - raw_reward)
        info["step_reward_clip_positive"] = raw_reward > reward
        info["step_reward_clip_negative"] = raw_reward < reward
        info["frames_run"] = int(step_result.summary.frames_run)
        info["repeat_index"] = max(step_result.summary.frames_run - 1, 0)
        info["energy_loss_total"] = float(step_result.summary.energy_loss_total)
        info["energy_gain_total"] = float(step_result.summary.energy_gain_total)
        info["damage_taken_frames"] = int(step_result.summary.damage_taken_frames)
        info["impact_frames"] = int(step_result.summary.impact_frames)
        info["airborne_frames"] = int(step_result.summary.airborne_frames)
        info["entered_state_flags"] = int(step_result.summary.entered_state_flags)
        info["entered_state_labels"] = tuple(step_result.summary.entered_state_labels)
        info["entered_crashed"] = bool(step_result.summary.entered_crashed)
        info["entered_retired"] = bool(step_result.summary.entered_retired)
        info["entered_finished"] = bool(step_result.summary.entered_finished)
        info["episode_airborne_frames"] = episode_airborne_frames
        info["boost_pad_entered"] = boost_pad_entered
        info["gas_level"] = gas_level
        info["gas_used"] = gas_level > 0.0
        info["air_brake_used"] = air_brake_used
        info["boost_used"] = boost_used
        info["lean_used"] = lean_used
        info["spin_requested"] = spin_requested
        info["spin_started"] = spin_started
        info["spin_macro_active_frames"] = spin_active_frames
        info["lean_macro_owned_frames"] = int(step_result.summary.lean_macro_owned_frames)
        info["spin_macro_active"] = bool(status.spin_macro_active)
        info["spin_macro_frames_remaining"] = int(status.spin_macro_frames_remaining)
        info["spin_macro_cooldown_frames"] = int(status.spin_macro_cooldown_frames)
        if reward_breakdown:
            info["reward_breakdown"] = reward_breakdown
        if reward_step.debug_info:
            info.update(reward_step.debug_info)
        info["episode_step"] = status.step_count
        info["stalled_steps"] = status.stalled_steps
        info["reverse_timer"] = status.reverse_timer
        info["progress_frontier_stalled_frames"] = status.progress_frontier_stalled_frames
        info["termination_reason"] = status.termination_reason
        info["truncation_reason"] = status.truncation_reason
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
            requested_control_state=requested_control_state,
            frames_run=step_result.summary.frames_run,
            gas_level=gas_level,
        )
        self.mask_controller.set_lean_allowed_values(
            ((0,) if status.spin_macro_active else self.control_state.lean_action_mask_override()),
        )
        self.mask_controller.set_spin_allowed_values(
            (0,) if status.spin_macro_active or status.spin_macro_cooldown_frames > 0 else None,
        )
        sync_dynamic_action_masks(
            mask_controller=self.mask_controller,
            control_state=self.control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self.config.boost_min_energy_fraction,
            mask_boost_when_active=self.action_config.mask_boost_when_active,
            mask_boost_when_airborne=self.action_config.mask_boost_when_airborne,
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
                display_controller_masks=step_result.display_controller_masks,
            ),
            telemetry=telemetry,
            requested_control_state=requested_control_state,
            gas_level=gas_level,
            episode_frame_count=status.step_count,
            episode_stalled_steps=status.stalled_steps,
            episode_progress_frontier_stalled_frames=(status.progress_frontier_stalled_frames),
            episode_progress_frontier_distance=episode_status.progress_frontier_distance,
            episode_progress_frontier_initialized=(episode_status.progress_frontier_initialized),
            episode_return=episode_return,
            episode_boost_pad_entries=episode_boost_pad_entries,
            episode_airborne_frames=episode_airborne_frames,
        )

    def _native_step(
        self,
        control_state: RaceControlState,
        request: EnvStepRequest,
    ) -> BackendStepResult:
        lean_timer_assist = self.action_config.lean_mode == "timer_assist"
        if request.capture_display_frames:
            return self.backend.step_repeat_watch_raw(
                control_state,
                action_repeat=request.action_repeat,
                frame_stack=self.config.observation.frame_stack,
                stack_mode=self.config.observation.stack_mode,
                minimap_layer=self.config.observation.minimap_layer,
                resize_filter=self.config.observation.resize_filter,
                minimap_resize_filter=self.config.observation.minimap_resize_filter,
                stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
                energy_loss_epsilon=self.reward_summary_config.energy_loss_epsilon,
                max_episode_steps=self.config.max_episode_steps,
                progress_frontier_stall_limit_frames=(
                    self.config.progress_frontier_stall_limit_frames
                ),
                progress_frontier_epsilon=float(self.config.progress_frontier_epsilon),
                terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
                lean_timer_assist=lean_timer_assist,
                spin_request=request.spin_request,
                spin_cooldown_frames=self.action_config.spin_cooldown_frames,
                **self.config.observation.native_resolution_kwargs(renderer=self.renderer),
            )
        return self.backend.step_repeat_raw(
            control_state,
            action_repeat=request.action_repeat,
            frame_stack=self.config.observation.frame_stack,
            stack_mode=self.config.observation.stack_mode,
            minimap_layer=self.config.observation.minimap_layer,
            resize_filter=self.config.observation.resize_filter,
            minimap_resize_filter=self.config.observation.minimap_resize_filter,
            stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
            energy_loss_epsilon=self.reward_summary_config.energy_loss_epsilon,
            max_episode_steps=self.config.max_episode_steps,
            progress_frontier_stall_limit_frames=(self.config.progress_frontier_stall_limit_frames),
            progress_frontier_epsilon=float(self.config.progress_frontier_epsilon),
            terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
            lean_timer_assist=lean_timer_assist,
            spin_request=request.spin_request,
            spin_cooldown_frames=self.action_config.spin_cooldown_frames,
            **self.config.observation.native_resolution_kwargs(renderer=self.renderer),
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


def _native_episode_status(
    *,
    request: EnvStepRequest,
    step_result: BackendStepResult,
    config: EnvConfig,
) -> _EpisodeStepStatus:
    """Project native env status while keeping shared episode state current."""

    telemetry = step_result.telemetry
    _frontier_stalled_frames, frontier_distance, frontier_initialized = _episode_progress_frontier(
        request=request,
        step_result=step_result,
        config=config,
        in_race_mode=bool(telemetry is not None and telemetry.in_race_mode),
    )
    return _EpisodeStepStatus(
        status=step_result.status,
        progress_frontier_distance=frontier_distance,
        progress_frontier_initialized=frontier_initialized,
    )


def _live_race_episode_status(
    *,
    request: EnvStepRequest,
    step_result: BackendStepResult,
    config: EnvConfig,
) -> _EpisodeStepStatus:
    """Build live-race status from shared Python episode state."""

    frames_run = int(step_result.summary.frames_run)
    telemetry = step_result.telemetry
    in_race_mode = bool(telemetry is not None and telemetry.in_race_mode)
    frontier_stalled_frames, frontier_distance, frontier_initialized = _episode_progress_frontier(
        request=request,
        step_result=step_result,
        config=config,
        in_race_mode=in_race_mode,
    )
    step_count = request.episode_frame_count + frames_run
    status_payload: StepStatusDict = {
        "step_count": step_count,
        "stalled_steps": _carried_stalled_steps(
            previous=request.episode_stalled_steps,
            trailing_in_step=int(step_result.summary.consecutive_low_speed_frames),
            frames_run=frames_run,
            in_race_mode=in_race_mode,
        ),
        "reverse_timer": step_result.status.reverse_timer,
        "progress_frontier_stalled_frames": frontier_stalled_frames,
        "termination_reason": step_result.status.termination_reason,
        "spin_macro_active": step_result.status.spin_macro_active,
        "spin_macro_frames_remaining": step_result.status.spin_macro_frames_remaining,
        "spin_macro_cooldown_frames": step_result.status.spin_macro_cooldown_frames,
    }
    return _EpisodeStepStatus(
        status=StepStatus(status_payload),
        progress_frontier_distance=frontier_distance,
        progress_frontier_initialized=frontier_initialized,
    )


def _carried_stalled_steps(
    *,
    previous: int,
    trailing_in_step: int,
    frames_run: int,
    in_race_mode: bool,
) -> int:
    if not in_race_mode or trailing_in_step == 0:
        return 0
    if trailing_in_step == frames_run:
        return previous + trailing_in_step
    return trailing_in_step


def _episode_progress_frontier(
    *,
    request: EnvStepRequest,
    step_result: BackendStepResult,
    config: EnvConfig,
    in_race_mode: bool,
) -> tuple[int, float, bool]:
    if not in_race_mode or int(step_result.summary.frames_run) == 0:
        return 0, 0.0, False

    max_race_distance = float(step_result.summary.max_race_distance)
    if not request.episode_progress_frontier_initialized:
        return 0, max_race_distance, True

    frontier_epsilon = float(config.progress_frontier_epsilon)
    if max_race_distance >= request.episode_progress_frontier_distance + frontier_epsilon:
        return 0, max_race_distance, True

    return (
        request.episode_progress_frontier_stalled_frames + int(step_result.summary.frames_run),
        request.episode_progress_frontier_distance,
        True,
    )
