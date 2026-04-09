# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces

from fzerox_emulator import (
    ControllerState,
    EmulatorBackend,
    FZeroXTelemetry,
    ObservationSpec,
)
from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, RewardConfig
from rl_fzerox.core.envs.actions import (
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    ActionValue,
    build_action_adapter,
)
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.observations import (
    DRIFT_DOUBLE_TAP_WINDOW_FRAMES,
    RECENT_BOOST_PRESSURE_WINDOW_FRAMES,
    STATE_FEATURE_COUNT,
    STATE_FEATURE_NAMES,
    ObservationValue,
    build_observation,
    build_observation_space,
    image_observation_shape,
)
from rl_fzerox.core.envs.rewards import RewardActionContext, build_reward_tracker
from rl_fzerox.core.seed import derive_seed

_DOMAIN_RESET_RNG = 0xD6E8_2BC9_2A5F_1873


class FZeroXEnvEngine:
    """Environment step engine around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply reward shaping
    and Gym-facing info assembly.
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
        self._action_adapter = build_action_adapter(config.action)
        self._curriculum_config = curriculum_config
        self._observation_spec = backend.observation_spec(config.observation.preset)
        self._reward_tracker = build_reward_tracker(
            config=reward_config,
            max_episode_steps=config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self._last_info: dict[str, object] = {}
        self._reset_count = 0
        self._rng_seed_base: int | None = None
        self._recent_boost_frames: deque[int] = deque(
            maxlen=RECENT_BOOST_PRESSURE_WINDOW_FRAMES,
        )
        self._recent_boost_frame_sum = 0
        self._left_drift_held = False
        self._right_drift_held = False
        self._left_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES
        self._right_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES
        self._action_space = self._action_adapter.action_space
        self._observation_space = build_observation_space(
            self._observation_spec,
            frame_stack=config.observation.frame_stack,
            mode=config.observation.mode,
        )
        self._base_action_mask_overrides = (
            config.action.mask.branch_overrides() if config.action.mask is not None else None
        )
        self._curriculum_stage_masks = _curriculum_stage_masks(curriculum_config)
        self._curriculum_stage_names = _curriculum_stage_names(curriculum_config)
        self._curriculum_stage_index = 0 if self._curriculum_stage_masks else None
        self._boost_unlocked: bool | None = None

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def action_masks(self) -> np.ndarray:
        """Return the flattened boolean action mask for the current stage."""

        stage_overrides = None
        if self._curriculum_stage_index is not None:
            stage_overrides = self._curriculum_stage_masks[self._curriculum_stage_index]
        return self._action_adapter.action_mask(
            base_overrides=self._base_action_mask_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=_dynamic_action_mask_overrides(boost_unlocked=self._boost_unlocked),
        )

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        if not self._curriculum_stage_masks:
            raise RuntimeError("No curriculum stages are configured for this env")
        if not 0 <= stage_index < len(self._curriculum_stage_masks):
            raise ValueError(f"Invalid curriculum stage index {stage_index}")
        self._curriculum_stage_index = int(stage_index)

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._curriculum_stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        if self._curriculum_stage_index is None:
            return None
        return self._curriculum_stage_names[self._curriculum_stage_index]

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode.

        The `seed` is kept for Gym compatibility and future Python-side reset
        randomization. The emulator baseline reset path itself is deterministic
        today, so different seeds currently do not diversify emulator state.
        """

        _, info, telemetry = self._reset_race_state()
        if seed is not None:
            self._rng_seed_base = seed
        telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
        self._boost_unlocked = _telemetry_can_boost(telemetry)
        self._reward_tracker.reset(telemetry)
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self._clear_recent_boost_pressure()
        self._reset_drift_press_state()
        info["seed"] = seed
        _set_curriculum_info(
            info,
            stage_index=self._curriculum_stage_index,
            stage_names=self._curriculum_stage_names,
        )
        if telemetry is not None:
            info.update(_telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        image_observation = self._render_observation_image()
        observation = build_observation(
            image=image_observation,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            left_drift_held=float(self._left_drift_held),
            right_drift_held=float(self._right_drift_held),
            left_press_age_norm=self._drift_press_age_norm(self._left_press_age_frames),
            right_press_age_norm=self._drift_press_age_norm(self._right_press_age_frames),
            recent_boost_pressure=self._recent_boost_pressure(),
        )
        _set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
        )
        self._last_info = dict(info)
        self._reset_count += 1
        return observation, info

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.step_control(self._action_adapter.decode(action))

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        """Decode one policy action into the held controller state it represents."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: ControllerState,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        self._held_controller_state = control_state
        return self._run_env_step(
            control_state,
            action_repeat=self.config.action_repeat,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        if control_state is not None:
            self._held_controller_state = control_state
        return self._run_env_step(
            self._held_controller_state,
            action_repeat=1,
        )

    def render(self) -> np.ndarray:
        return self.backend.render_display(preset=self.config.observation.preset)

    def close(self) -> None:
        self.backend.close()

    def _reset_race_state(self) -> tuple[np.ndarray, dict[str, object], FZeroXTelemetry | None]:
        continue_error: str | None = None
        if self.config.reset_to_race and not _has_custom_baseline(self._last_info):
            if self._episode_done:
                try:
                    frame, reset_info = continue_to_next_race(self.backend)
                    info = _reset_context_info(self._last_info)
                    info.update(reset_info)
                    return frame, info, _read_live_telemetry(self.backend)
                except RuntimeError as exc:
                    continue_error = str(exc)

        reset_state = self.backend.reset()
        info = dict(reset_state.info)
        frame = reset_state.frame
        if continue_error is not None:
            info["reset_fallback"] = "continue_to_next_race_failed"
            info["continue_to_next_race_error"] = continue_error

        if self.config.reset_to_race and not _has_custom_baseline(info):
            frame, boot_info = boot_into_first_race(self.backend)
            info.update(boot_info)
            return frame, info, _read_live_telemetry(self.backend)

        return frame, info, _read_live_telemetry(self.backend)

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
        return _read_live_telemetry(self.backend) or telemetry

    def _run_env_step(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        step_result = self.backend.step_repeat_raw(
            control_state,
            action_repeat=action_repeat,
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
            stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
            energy_loss_epsilon=self._reward_summary_config.energy_loss_epsilon,
            max_episode_steps=self.config.max_episode_steps,
            stuck_step_limit=self.config.stuck_step_limit,
            wrong_way_timer_limit=self.config.wrong_way_timer_limit,
            terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
        )
        info = _backend_step_info(self.backend)
        telemetry = step_result.telemetry
        self._boost_unlocked = _telemetry_can_boost(telemetry)
        reward_step = self._reward_tracker.step_summary(
            step_result.summary,
            step_result.status,
            telemetry,
            RewardActionContext(
                boost_requested=bool(control_state.joypad_mask & BOOST_MASK),
            ),
        )
        reward = reward_step.reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = step_result.status.terminated
        truncated = step_result.status.truncated
        info["step_reward"] = reward
        info["repeat_index"] = max(step_result.summary.frames_run - 1, 0)
        if reward_breakdown:
            info["reward_breakdown"] = reward_breakdown
        info["episode_step"] = step_result.status.step_count
        info["stalled_steps"] = step_result.status.stalled_steps
        info["reverse_timer"] = step_result.status.reverse_timer
        info["termination_reason"] = step_result.status.termination_reason
        info["truncation_reason"] = step_result.status.truncation_reason
        _set_curriculum_info(
            info,
            stage_index=self._curriculum_stage_index,
            stage_names=self._curriculum_stage_names,
        )
        if telemetry is not None:
            # Keep env info pickle-safe for SubprocVecEnv workers.
            info.update(_telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        self._record_recent_boost_pressure(
            boost_requested=bool(control_state.joypad_mask & BOOST_MASK),
            frames_run=step_result.summary.frames_run,
        )
        self._update_drift_press_state(
            control_state=control_state,
            frames_run=step_result.summary.frames_run,
        )
        image_observation = step_result.observation
        observation = build_observation(
            image=image_observation,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            left_drift_held=float(self._left_drift_held),
            right_drift_held=float(self._right_drift_held),
            left_press_age_norm=self._drift_press_age_norm(self._left_press_age_frames),
            right_press_age_norm=self._drift_press_age_norm(self._right_press_age_frames),
            recent_boost_pressure=self._recent_boost_pressure(),
        )
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        _set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
        )
        return observation, reward, terminated, truncated, info

    def _render_observation_image(self) -> np.ndarray:
        return self.backend.render_observation(
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
        )

    def _clear_recent_boost_pressure(self) -> None:
        self._recent_boost_frames.clear()
        self._recent_boost_frame_sum = 0

    def _record_recent_boost_pressure(self, *, boost_requested: bool, frames_run: int) -> None:
        encoded = 1 if boost_requested else 0
        for _ in range(max(int(frames_run), 0)):
            if len(self._recent_boost_frames) == self._recent_boost_frames.maxlen:
                removed = self._recent_boost_frames.popleft()
                self._recent_boost_frame_sum -= removed
            self._recent_boost_frames.append(encoded)
            self._recent_boost_frame_sum += encoded

    def _recent_boost_pressure(self) -> float:
        if not self._recent_boost_frames:
            return 0.0
        return self._recent_boost_frame_sum / len(self._recent_boost_frames)

    def _reset_drift_press_state(self) -> None:
        self._left_drift_held = False
        self._right_drift_held = False
        self._left_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES
        self._right_press_age_frames = DRIFT_DOUBLE_TAP_WINDOW_FRAMES

    def _update_drift_press_state(
        self,
        *,
        control_state: ControllerState,
        frames_run: int,
    ) -> None:
        frames_elapsed = max(int(frames_run), 0)
        left_held = bool(control_state.joypad_mask & DRIFT_LEFT_MASK)
        right_held = bool(control_state.joypad_mask & DRIFT_RIGHT_MASK)
        self._left_press_age_frames = _advance_press_age(
            self._left_press_age_frames,
            was_held=self._left_drift_held,
            is_held=left_held,
            frames_elapsed=frames_elapsed,
        )
        self._right_press_age_frames = _advance_press_age(
            self._right_press_age_frames,
            was_held=self._right_drift_held,
            is_held=right_held,
            frames_elapsed=frames_elapsed,
        )
        self._left_drift_held = left_held
        self._right_drift_held = right_held

    def _drift_press_age_norm(self, frames: int) -> float:
        clamped_frames = min(max(int(frames), 0), DRIFT_DOUBLE_TAP_WINDOW_FRAMES)
        return clamped_frames / DRIFT_DOUBLE_TAP_WINDOW_FRAMES


def _has_custom_baseline(info: dict[str, object]) -> bool:
    baseline_kind = info.get("baseline_kind")
    return baseline_kind == "custom"


def _read_live_telemetry(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    return backend.try_read_telemetry()


def _advance_press_age(
    previous_age_frames: int,
    *,
    was_held: bool,
    is_held: bool,
    frames_elapsed: int,
) -> int:
    if is_held and not was_held:
        return min(frames_elapsed, DRIFT_DOUBLE_TAP_WINDOW_FRAMES)
    return min(previous_age_frames + frames_elapsed, DRIFT_DOUBLE_TAP_WINDOW_FRAMES)


def _set_observation_info(
    info: dict[str, object],
    *,
    observation_shape: tuple[int, ...],
    observation_spec: ObservationSpec,
    frame_stack: int,
    observation_mode: str,
) -> None:
    expected_image_shape = image_observation_shape(observation_spec, frame_stack=frame_stack)
    if observation_shape != expected_image_shape:
        raise ValueError(
            "Rendered observation shape did not match native observation spec: "
            f"got={observation_shape}, expected={expected_image_shape}"
        )

    info["observation_mode"] = observation_mode
    info["observation_shape"] = observation_shape
    info["observation_frame_shape"] = (
        observation_spec.height,
        observation_spec.width,
        observation_spec.channels,
    )
    info["observation_stack"] = frame_stack
    if observation_mode == "image_state":
        info["observation_state_shape"] = (STATE_FEATURE_COUNT,)
        info["observation_state_features"] = STATE_FEATURE_NAMES


def _reset_context_info(info: dict[str, object]) -> dict[str, object]:
    keys = (
        "backend",
        "core_path",
        "rom_path",
        "runtime_dir",
        "baseline_state_path",
        "baseline_kind",
        "display_aspect_ratio",
        "native_fps",
    )
    return {key: info[key] for key in keys if key in info}


def _telemetry_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    race_laps_completed = completed_race_laps(telemetry)
    return {
        "game_mode": telemetry.game_mode_name,
        "course_index": telemetry.course_index,
        "race_time_ms": telemetry.player.race_time_ms,
        "race_distance": telemetry.player.race_distance,
        "speed_kph": telemetry.player.speed_kph,
        "position": telemetry.player.position,
        "lap": telemetry.player.lap,
        "laps_completed": race_laps_completed,
        "race_laps_completed": race_laps_completed,
        "raw_laps_completed": telemetry.player.laps_completed,
        "energy": telemetry.player.energy,
    }


def _telemetry_can_boost(telemetry: FZeroXTelemetry | None) -> bool:
    if telemetry is None:
        return False
    return bool(telemetry.player.can_boost)


def _dynamic_action_mask_overrides(
    *,
    boost_unlocked: bool | None,
) -> dict[str, tuple[int, ...]] | None:
    if boost_unlocked is not False:
        return None
    return {"boost": (0,)}


def _backend_step_info(backend: EmulatorBackend) -> dict[str, object]:
    return {
        "backend": backend.name,
        "frame_index": backend.frame_index,
        "display_aspect_ratio": backend.display_aspect_ratio,
        "native_fps": backend.native_fps,
    }


def _curriculum_stage_masks(
    curriculum_config: CurriculumConfig | None,
) -> tuple[dict[str, tuple[int, ...]] | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(
        stage.action_mask.branch_overrides() if stage.action_mask is not None else None
        for stage in curriculum_config.stages
    )


def _curriculum_stage_names(
    curriculum_config: CurriculumConfig | None,
) -> tuple[str, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.name for stage in curriculum_config.stages)


def _set_curriculum_info(
    info: dict[str, object],
    *,
    stage_index: int | None,
    stage_names: tuple[str, ...],
) -> None:
    info["curriculum_stage"] = stage_index
    if stage_index is None:
        info["curriculum_stage_name"] = None
        return
    info["curriculum_stage_name"] = stage_names[stage_index]
