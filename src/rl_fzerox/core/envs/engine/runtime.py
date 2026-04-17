# src/rl_fzerox/core/envs/engine/runtime.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState, EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.boot import (
    boot_into_first_race,
    continue_to_next_race,
    sync_race_intro_target,
)
from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)
from rl_fzerox.core.domain.hybrid_action import HYBRID_CONTINUOUS_ACTION_KEY
from rl_fzerox.core.domain.lean import LEAN_MODE_TIMER_ASSIST
from rl_fzerox.core.envs.actions import (
    AIR_BRAKE_MASK,
    BOOST_MASK,
    ActionValue,
    ResettableActionAdapter,
    build_action_adapter,
)
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.observations import (
    ObservationValue,
    build_observation,
    build_observation_space,
)
from rl_fzerox.core.envs.rewards import RewardActionContext, build_reward_tracker
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.seed import derive_seed

from .camera import sync_camera_setting
from .control_state import ControlStateTracker
from .info import (
    backend_step_info,
    has_custom_baseline,
    read_live_telemetry,
    reset_context_info,
    set_curriculum_info,
    set_observation_info,
    telemetry_can_boost,
    telemetry_energy_fraction,
    telemetry_info,
)
from .masks import ActionMaskController
from .tracks import SelectedTrack, TrackBaselineCache, select_reset_track

_DOMAIN_RESET_RNG = 0xD6E8_2BC9_2A5F_1873
_DOMAIN_REWARD_MILESTONE_PHASE = 0xA409_3822_299F_31D0
_DOMAIN_TRACK_SAMPLING = 0x35E7_40D8_FF53_42B1


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
    ) -> None:
        self.backend = backend
        self.config = config
        self._curriculum_config = curriculum_config
        self._action_adapter = build_action_adapter(config.action)
        self._observation_spec = backend.observation_spec(config.observation.preset)
        self._reward_tracker = build_reward_tracker(
            config=reward_config,
            max_episode_steps=config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._action_space = self._action_adapter.action_space
        self._observation_space = build_observation_space(
            self._observation_spec,
            frame_stack=config.observation.frame_stack,
            mode=config.observation.mode,
            state_profile=config.observation.state_profile,
            action_history_len=config.observation.action_history_len,
            action_history_controls=config.observation.action_history_controls,
        )
        self._mask_controller = ActionMaskController.from_config(
            adapter=self._action_adapter,
            base_overrides=(
                config.action.mask.branch_overrides() if config.action.mask is not None else None
            ),
            curriculum_config=curriculum_config,
            boost_unmask_max_speed_kph=config.action.boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=config.action.lean_unmask_min_speed_kph,
        )
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )
        self._track_baseline_cache = TrackBaselineCache()
        self._control_state = ControlStateTracker(
            lean_mode=config.action.lean_mode,
            boost_decision_interval_frames=config.action.boost_decision_interval_frames,
            boost_request_lockout_frames=config.action.boost_request_lockout_frames,
            action_history_len=config.observation.action_history_len,
            action_history_controls=config.observation.action_history_controls,
        )
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self._last_info: dict[str, object] = {}
        self._last_telemetry: FZeroXTelemetry | None = None
        self._manual_boost_allowed: bool | None = None
        self._reset_count = 0
        self._rng_seed_base: int | None = None

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def action_masks(self) -> np.ndarray:
        """Return the flattened boolean action mask for the current stage."""

        return self._mask_controller.action_mask()

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        self._mask_controller.set_curriculum_stage(stage_index)
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        """Align watch-time stage masks with the loaded checkpoint metadata."""

        self._mask_controller.sync_checkpoint_stage(stage_index)
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._mask_controller.stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        return self._mask_controller.stage_name

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode and return the first policy observation."""

        if seed is not None:
            self._rng_seed_base = seed
        if self.config.benchmark_noop_reset:
            selected_track = None
            info, telemetry = self._benchmark_noop_reset_state()
        else:
            selected_track = self._select_reset_track(seed)
            if selected_track is not None:
                if self.config.cache_track_baselines:
                    self._track_baseline_cache.load_into_backend(
                        self.backend,
                        selected_track.baseline_state_path,
                    )
                else:
                    self.backend.load_baseline(selected_track.baseline_state_path)
            _, info, telemetry = self._reset_race_state()
            if selected_track is not None:
                info.update(selected_track.info())
            telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
            telemetry = sync_camera_setting(
                self.backend,
                target_name=self.config.camera_setting,
                telemetry=telemetry,
                info=info,
            )
            race_intro_info, telemetry = sync_race_intro_target(
                self.backend,
                target_timer=self.config.race_intro_target_timer,
            )
            info.update(race_intro_info)
            info.update(backend_step_info(self.backend))
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self.backend.set_controller_state(self._held_controller_state)
        self._control_state.reset()
        self._mask_controller.set_lean_allowed_values(None)
        self._sync_dynamic_masks(telemetry)
        self._last_telemetry = telemetry
        self._reward_tracker.reset(
            telemetry,
            episode_seed=self._reward_episode_seed(seed),
        )
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
        image_observation = self._render_observation_image()
        observation = self._build_observation(image=image_observation, telemetry=telemetry)
        set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
            observation_state_profile=self.config.observation.state_profile,
            action_history_len=self.config.observation.action_history_len,
            action_history_controls=self.config.observation.action_history_controls,
        )
        self._last_info = dict(info)
        self._reset_count += 1
        return observation, info

    def _benchmark_noop_reset_state(self) -> tuple[dict[str, object], FZeroXTelemetry | None]:
        """Reset Python episode bookkeeping without restoring emulator state.

        This is only for throughput diagnostics. It preserves the live game
        state, so it is not a valid training reset for real learning runs.
        """

        info = backend_step_info(self.backend)
        info["reset_mode"] = "benchmark_noop_reset"
        info["benchmark_noop_reset"] = True
        return info, read_live_telemetry(self.backend)

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_control_state(
            self._action_adapter.decode(action),
            action_drive_axis=_action_drive_axis(action, self._action_space),
        )

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        """Decode one policy action into the held controller state it represents."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: ControllerState,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_control_state(control_state, action_drive_axis=None)

    def _step_control_state(
        self,
        control_state: ControllerState,
        *,
        action_drive_axis: float | None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._held_controller_state = applied_control_state
        return self._run_env_step(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        requested_control_state = (
            self._held_controller_state if control_state is None else control_state
        )
        self._held_controller_state = self._apply_control_semantics(requested_control_state)
        return self._run_env_step(
            self._held_controller_state,
            action_repeat=1,
            requested_control_state=requested_control_state,
            action_drive_axis=None,
        )

    def render(self) -> np.ndarray:
        return self.backend.render_display(preset=self.config.observation.preset)

    def close(self) -> None:
        self.backend.close()

    def _reset_race_state(self) -> tuple[np.ndarray, dict[str, object], FZeroXTelemetry | None]:
        continue_error: str | None = None
        if self.config.reset_to_race and not has_custom_baseline(self._last_info):
            if self._episode_done:
                try:
                    frame, reset_info = continue_to_next_race(self.backend)
                    info = reset_context_info(self._last_info)
                    info.update(reset_info)
                    return frame, info, read_live_telemetry(self.backend)
                except RuntimeError as exc:
                    continue_error = str(exc)

        reset_state = self.backend.reset()
        info = dict(reset_state.info)
        frame = reset_state.frame
        if continue_error is not None:
            info["reset_fallback"] = "continue_to_next_race_failed"
            info["continue_to_next_race_error"] = continue_error

        if self.config.reset_to_race and not has_custom_baseline(info):
            frame, boot_info = boot_into_first_race(self.backend)
            info.update(boot_info)
            return frame, info, read_live_telemetry(self.backend)

        return frame, info, read_live_telemetry(self.backend)

    def _maybe_randomize_game_rng(
        self,
        seed: int | None,
        telemetry: FZeroXTelemetry | None,
        info: dict[str, object],
    ) -> FZeroXTelemetry | None:
        if not self.config.randomize_game_rng_on_reset:
            return telemetry
        seed_base = seed if seed is not None else self._rng_seed_base
        if seed_base is None:
            return telemetry
        if self.config.randomize_game_rng_requires_race_mode and (
            telemetry is None or not telemetry.in_race_mode
        ):
            info["rng_randomized"] = False
            info["rng_randomization_skip_reason"] = "not_in_race"
            return telemetry

        rng_seed = derive_seed(seed_base, _DOMAIN_RESET_RNG, self._reset_count)
        if rng_seed is None:
            return telemetry
        rng_state = self.backend.randomize_game_rng(rng_seed)
        info["rng_randomized"] = True
        info["rng_seed"] = rng_seed
        info["rng_state"] = rng_state
        return read_live_telemetry(self.backend) or telemetry

    def _reward_episode_seed(self, seed: int | None) -> int | None:
        seed_base = seed if seed is not None else self._rng_seed_base
        if seed_base is None:
            return None
        return derive_seed(seed_base, _DOMAIN_REWARD_MILESTONE_PHASE, self._reset_count)

    def _select_reset_track(self, seed: int | None) -> SelectedTrack | None:
        seed_base = seed if seed is not None else self._rng_seed_base
        sampling_seed = (
            None
            if seed_base is None
            else derive_seed(seed_base, _DOMAIN_TRACK_SAMPLING, self._reset_count)
        )
        return select_reset_track(self._active_track_sampling, seed=sampling_seed)

    def _stage_track_sampling_config(self, stage_index: int | None) -> TrackSamplingConfig:
        if (
            self._curriculum_config is None
            or not self._curriculum_config.enabled
            or stage_index is None
        ):
            return self.config.track_sampling
        stage = self._curriculum_config.stages[stage_index]
        return stage.track_sampling or self.config.track_sampling

    def _sync_dynamic_masks(self, telemetry: FZeroXTelemetry | None) -> None:
        if telemetry is None:
            self._manual_boost_allowed = None
            self._mask_controller.set_boost_unlocked(None)
            self._mask_controller.set_speed_kph(None)
            return
        speed_kph = float(telemetry.player.speed_kph)
        self._mask_controller.set_speed_kph(speed_kph)
        can_boost = telemetry_can_boost(telemetry)
        can_boost = can_boost and not telemetry_boost_active(telemetry)
        can_boost = can_boost and not telemetry.player.airborne
        can_boost = can_boost and telemetry.player.reverse_timer <= 0
        can_boost = can_boost and self._control_state.boost_action_allowed_by_timing()
        max_boost_speed = self.config.action.boost_unmask_max_speed_kph
        if max_boost_speed is not None:
            can_boost = can_boost and speed_kph < float(max_boost_speed)
        energy_fraction = telemetry_energy_fraction(telemetry)
        min_energy_fraction = float(self.config.boost_min_energy_fraction)
        if energy_fraction is not None and min_energy_fraction > 0.0:
            can_boost = can_boost and energy_fraction >= min_energy_fraction
        self._manual_boost_allowed = can_boost
        self._mask_controller.set_boost_unlocked(can_boost)

    def _run_env_step(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
        requested_control_state: ControllerState | None = None,
        action_drive_axis: float | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        requested_control_state = requested_control_state or control_state
        applied_control_state = control_state
        stuck_step_limit = (
            self.config.stuck_step_limit
            if self.config.stuck_truncation_enabled
            else self.config.max_episode_steps + 1
        )
        step_result = self.backend.step_repeat_raw(
            applied_control_state,
            action_repeat=action_repeat,
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
            stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
            energy_loss_epsilon=self._reward_summary_config.energy_loss_epsilon,
            max_episode_steps=self.config.max_episode_steps,
            stuck_step_limit=stuck_step_limit,
            wrong_way_timer_limit=(
                self.config.wrong_way_timer_limit
                if self.config.wrong_way_truncation_enabled
                else None
            ),
            progress_frontier_stall_limit_frames=self.config.progress_frontier_stall_limit_frames,
            progress_frontier_epsilon=float(self.config.progress_frontier_epsilon),
            terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
            lean_timer_assist=(
                self.config.action.lean_mode == LEAN_MODE_TIMER_ASSIST
            ),
        )
        info = backend_step_info(self.backend)
        telemetry = step_result.telemetry
        self._last_telemetry = telemetry
        reward_step = self._reward_tracker.step_summary(
            step_result.summary,
            step_result.status,
            telemetry,
            RewardActionContext(
                air_brake_requested=bool(requested_control_state.joypad_mask & AIR_BRAKE_MASK),
                boost_requested=bool(applied_control_state.joypad_mask & BOOST_MASK),
                drive_axis=action_drive_axis,
            ),
        )
        reward = reward_step.reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = step_result.status.terminated
        truncated = step_result.status.truncated
        info["step_reward"] = reward
        info["repeat_index"] = max(step_result.summary.frames_run - 1, 0)
        info["energy_loss_total"] = float(step_result.summary.energy_loss_total)
        info["damage_taken_frames"] = int(step_result.summary.damage_taken_frames)
        info["collision_recoil_entered"] = bool(step_result.summary.entered_collision_recoil)
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
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            # Keep env info pickle-safe for SubprocVecEnv workers.
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        self._control_state.record_step(
            control_state=applied_control_state,
            frames_run=step_result.summary.frames_run,
        )
        self._mask_controller.set_lean_allowed_values(
            self._control_state.lean_action_mask_override(),
        )
        self._sync_dynamic_masks(telemetry)
        image_observation = step_result.observation
        observation = self._build_observation(image=image_observation, telemetry=telemetry)
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
            observation_state_profile=self.config.observation.state_profile,
            action_history_len=self.config.observation.action_history_len,
            action_history_controls=self.config.observation.action_history_controls,
        )
        return observation, reward, terminated, truncated, info

    def _apply_control_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply telemetry gates and configured lean semantics."""

        gated_control_state = self._apply_dynamic_control_gates(control_state)
        return self._control_state.apply_lean_semantics(gated_control_state)

    def _apply_dynamic_control_gates(self, control_state: ControllerState) -> ControllerState:
        """Suppress controls whose validity depends on the latest telemetry."""

        if self._manual_boost_allowed is False:
            control_state = _without_joypad_mask(control_state, BOOST_MASK)
        air_brake_mode = self.config.action.continuous_air_brake_mode
        if air_brake_mode == "always":
            return control_state
        if air_brake_mode == "off":
            return _without_joypad_mask(control_state, AIR_BRAKE_MASK)
        if self._last_telemetry is None or self._last_telemetry.player.airborne:
            return control_state
        return _without_joypad_mask(control_state, AIR_BRAKE_MASK)

    def _render_observation_image(self) -> np.ndarray:
        return self.backend.render_observation(
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
        )

    def _build_observation(
        self,
        *,
        image: np.ndarray,
        telemetry: FZeroXTelemetry | None,
    ) -> ObservationValue:
        """Build the policy observation from the rendered image plus control context."""

        return build_observation(
            image=image,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            state_profile=self.config.observation.state_profile,
            action_history_len=self.config.observation.action_history_len,
            action_history_controls=self.config.observation.action_history_controls,
            action_history=self._control_state.action_history_fields(),
            **self._control_state.observation_fields(),
        )


def _without_joypad_mask(control_state: ControllerState, joypad_mask: int) -> ControllerState:
    if not control_state.joypad_mask & joypad_mask:
        return control_state
    return ControllerState(
        joypad_mask=control_state.joypad_mask & ~joypad_mask,
        left_stick_x=control_state.left_stick_x,
        left_stick_y=control_state.left_stick_y,
        right_stick_x=control_state.right_stick_x,
        right_stick_y=control_state.right_stick_y,
    )


def _action_drive_axis(action: ActionValue, action_space: spaces.Space) -> float | None:
    """Extract the raw continuous drive axis when the active action space has one."""

    source: object
    if isinstance(action_space, spaces.Dict):
        if not isinstance(action, Mapping):
            return None
        source = action.get(HYBRID_CONTINUOUS_ACTION_KEY)
    elif isinstance(action_space, spaces.Box):
        source = action
    else:
        return None
    if source is None or isinstance(source, str | bytes):
        return None
    try:
        values = np.asarray(source, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 2 or not np.isfinite(values[1]):
        return None
    return float(np.clip(values[1], -1.0, 1.0))
