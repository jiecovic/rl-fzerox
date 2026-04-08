# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import (
    ControllerState,
    EmulatorBackend,
    FZeroXTelemetry,
    ObservationSpec,
)
from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.schema import EnvConfig, RewardConfig
from rl_fzerox.core.envs.actions import ActionValue, build_action_adapter
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.rewards import build_reward_tracker


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
    ) -> None:
        self.backend = backend
        self.config = config
        self._action_adapter = build_action_adapter(config.action)
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
        self._action_space = self._action_adapter.action_space
        self._observation_space = _observation_space(
            self._observation_spec,
            frame_stack=config.observation.frame_stack,
        )

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        """Reset one episode.

        The `seed` is kept for Gym compatibility and future Python-side reset
        randomization. The emulator baseline reset path itself is deterministic
        today, so different seeds currently do not diversify emulator state.
        """

        _, info = self._reset_race_state()
        telemetry = _read_live_telemetry(self.backend)
        self._reward_tracker.reset(telemetry)
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        info["seed"] = seed
        if telemetry is not None:
            info.update(_telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        observation = self._transform_observation()
        _set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
        )
        self._last_info = dict(info)
        return observation, info

    def step(
        self,
        action: ActionValue,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return self.step_control(self._action_adapter.decode(action))

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        """Decode one policy action into the held controller state it represents."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: ControllerState,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._held_controller_state = control_state
        return self._run_env_step(
            control_state,
            action_repeat=self.config.action_repeat,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
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

    def _reset_race_state(self) -> tuple[np.ndarray, dict[str, object]]:
        continue_error: str | None = None
        if self.config.reset_to_race and not _has_custom_baseline(self._last_info):
            if self._episode_done:
                try:
                    frame, reset_info = continue_to_next_race(self.backend)
                    info = _reset_context_info(self._last_info)
                    info.update(reset_info)
                    return frame, info
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

        return frame, info

    def _run_env_step(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
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
        )
        info = _backend_step_info(self.backend)
        telemetry = step_result.telemetry
        reward_step = self._reward_tracker.step_summary(
            step_result.summary,
            step_result.status,
            telemetry,
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
        if telemetry is not None:
            # Keep env info pickle-safe for SubprocVecEnv workers.
            info.update(_telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        observation = step_result.observation
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        _set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
        )
        return observation, reward, terminated, truncated, info

    def _transform_observation(self) -> np.ndarray:
        return self.backend.render_observation(
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
        )


def _has_custom_baseline(info: dict[str, object]) -> bool:
    baseline_kind = info.get("baseline_kind")
    return baseline_kind == "custom"


def _read_live_telemetry(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    return backend.try_read_telemetry()


def _observation_space(observation_spec: ObservationSpec, *, frame_stack: int) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=(
            observation_spec.height,
            observation_spec.width,
            observation_spec.channels * frame_stack,
        ),
        dtype=np.uint8,
    )


def _set_observation_info(
    info: dict[str, object],
    *,
    observation_shape: tuple[int, ...],
    observation_spec: ObservationSpec,
    frame_stack: int,
) -> None:
    info["observation_shape"] = observation_shape
    info["observation_frame_shape"] = (
        observation_spec.height,
        observation_spec.width,
        observation_spec.channels,
    )
    info["observation_stack"] = frame_stack


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
    return {
        "game_mode": telemetry.game_mode_name,
        "course_index": telemetry.course_index,
        "race_time_ms": telemetry.player.race_time_ms,
        "race_distance": telemetry.player.race_distance,
        "speed_kph": telemetry.player.speed_kph,
        "position": telemetry.player.position,
        "lap": telemetry.player.lap,
        "laps_completed": telemetry.player.laps_completed,
        "energy": telemetry.player.energy,
    }


def _backend_step_info(backend: EmulatorBackend) -> dict[str, object]:
    return {
        "backend": backend.name,
        "frame_index": backend.frame_index,
        "display_aspect_ratio": backend.display_aspect_ratio,
        "native_fps": backend.native_fps,
    }
