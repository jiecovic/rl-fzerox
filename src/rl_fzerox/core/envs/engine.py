# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions import ActionValue, SteerDriveActionAdapter
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.limits import EpisodeLimits
from rl_fzerox.core.envs.observations import FrameStackBuffer
from rl_fzerox.core.game import FZeroXTelemetry, RewardTracker


class FZeroXEnvEngine:
    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
    ) -> None:
        self.backend = backend
        self.config = config
        self._action_adapter = SteerDriveActionAdapter(config.action)
        self._observation_stack = FrameStackBuffer(
            frame_space=_observation_frame_space(config),
            frame_stack=config.observation.frame_stack,
        )
        self._episode_limits = EpisodeLimits(config)
        self._reward_tracker = RewardTracker()
        self._episode_done = False
        self._episode_return = 0.0
        self._last_info: dict[str, object] = {}
        self._action_space = self._action_adapter.action_space
        self._observation_space = self._observation_stack.observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        _, info = self._reset_race_state()
        telemetry = _read_live_telemetry(self.backend)
        self._reward_tracker.reset(telemetry)
        self._episode_limits.reset(telemetry)
        self._episode_done = False
        self._episode_return = 0.0
        info["seed"] = seed
        if telemetry is not None:
            info["telemetry"] = telemetry
        observation = self._reset_observation(info)
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
        self.backend.set_controller_state(control_state)
        latest_frame: np.ndarray | None = None
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, object] = {}
        last_telemetry: FZeroXTelemetry | None = None

        for repeat_index in range(self.config.action_repeat):
            is_last_repeat = repeat_index == (self.config.action_repeat - 1)
            (
                latest_frame,
                step_reward,
                terminated,
                truncated,
                info,
                last_telemetry,
            ) = self._advance_one_frame(render_frame=is_last_repeat)
            total_reward += step_reward
            info["repeat_index"] = repeat_index

            if terminated or truncated:
                break

        if latest_frame is None:
            raise RuntimeError("The emulator did not produce a frame during step()")

        if last_telemetry is not None:
            info["telemetry"] = last_telemetry
        self._episode_return += total_reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        observation = self._append_observation(info)

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            info,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        if control_state is not None:
            self.backend.set_controller_state(control_state)
        frame, reward, terminated, truncated, info, _ = self._advance_one_frame()
        if frame is None:
            raise RuntimeError("The emulator did not produce a frame during step_frame()")
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        observation = self._append_observation(info)
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()

    def _reset_race_state(self) -> tuple[np.ndarray, dict[str, object]]:
        if self.config.reset_to_race and not _has_custom_baseline(self._last_info):
            if self._episode_done:
                try:
                    frame, reset_info = continue_to_next_race(self.backend)
                    info = _reset_context_info(self._last_info)
                    info.update(reset_info)
                    return frame, info
                except RuntimeError:
                    pass

        reset_state = self.backend.reset()
        info = dict(reset_state.info)
        frame = reset_state.frame

        if self.config.reset_to_race and not _has_custom_baseline(info):
            frame, boot_info = boot_into_first_race(self.backend)
            info.update(boot_info)

        return frame, info

    def _advance_one_frame(
        self,
        *,
        render_frame: bool = True,
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, object], FZeroXTelemetry | None]:
        if render_frame:
            frame_step = self.backend.step_frame()
            frame = frame_step.frame
            frame_reward = frame_step.reward
            terminated = frame_step.terminated
            truncated = frame_step.truncated
            info = dict(frame_step.info)
        else:
            self.backend.step_frames(1)
            frame = None
            frame_reward = 0.0
            terminated = False
            truncated = False
            info = _backend_step_info(self.backend)
        telemetry = _read_live_telemetry(self.backend)
        reward_step = self._reward_tracker.step(telemetry)
        reward = frame_reward + reward_step.reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = terminated or reward_step.terminated
        limit_step = self._episode_limits.step(telemetry)
        truncation_reason = limit_step.truncation_reason
        truncated = truncated or (truncation_reason is not None)
        truncation_penalty, truncation_label = self._reward_tracker.truncation_penalty(
            truncation_reason
        )
        reward += truncation_penalty
        if truncation_label is not None:
            reward_breakdown[truncation_label] = truncation_penalty
        if frame is None and (terminated or truncated):
            frame = self.backend.render()
        info["step_reward"] = reward
        if reward_breakdown:
            info["reward_breakdown"] = reward_breakdown
        info["episode_step"] = limit_step.step_count
        info["stalled_steps"] = limit_step.stalled_steps
        info["reverse_steps"] = limit_step.reverse_steps
        if truncation_reason is not None:
            info["truncation_reason"] = truncation_reason
        if telemetry is not None:
            info["telemetry"] = telemetry
            info.update(_telemetry_info(telemetry))
            termination_reason = _termination_reason(telemetry)
            if termination_reason is not None:
                info["termination_reason"] = termination_reason
        return (
            frame,
            reward,
            terminated,
            truncated,
            info,
            telemetry,
        )

    def _reset_observation(self, info: dict[str, object]) -> np.ndarray:
        observation_frame = self._transform_observation()
        observation = self._observation_stack.reset(observation_frame)
        info["observation_shape"] = tuple(int(value) for value in observation.shape)
        info["observation_frame_shape"] = self._observation_stack.frame_shape
        info["observation_stack"] = self.config.observation.frame_stack
        return observation

    def _append_observation(self, info: dict[str, object]) -> np.ndarray:
        observation_frame = self._transform_observation()
        observation = self._observation_stack.append(observation_frame)
        info["observation_shape"] = tuple(int(value) for value in observation.shape)
        info["observation_frame_shape"] = self._observation_stack.frame_shape
        info["observation_stack"] = self.config.observation.frame_stack
        return observation

    def _transform_observation(self) -> np.ndarray:
        return self.backend.render_observation(
            width=self.config.observation.width,
            height=self.config.observation.height,
            rgb=self.config.observation.rgb,
        )

def _has_custom_baseline(info: dict[str, object]) -> bool:
    baseline_kind = info.get("baseline_kind")
    return baseline_kind == "custom"


def _read_live_telemetry(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    return backend.try_read_telemetry()


def _observation_frame_space(config: EnvConfig) -> spaces.Box:
    channels = 3 if config.observation.rgb else 1
    return spaces.Box(
        low=0,
        high=255,
        shape=(config.observation.height, config.observation.width, channels),
        dtype=np.uint8,
    )


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


def _termination_reason(telemetry: FZeroXTelemetry) -> str | None:
    terminal_labels = (
        "finished",
        "crashed",
        "retired",
        "falling_off_track",
    )
    for label in terminal_labels:
        if label in telemetry.player.state_labels:
            return label
    return None
