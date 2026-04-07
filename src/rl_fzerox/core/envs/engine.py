# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

from typing import cast

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions import ActionValue, SteerDriveActionAdapter
from rl_fzerox.core.envs.limits import EpisodeLimits
from rl_fzerox.core.envs.observations import FrameStackBuffer, ResizedObservationAdapter
from rl_fzerox.core.game import RewardTracker, read_telemetry
from rl_fzerox.core.game.telemetry import FZeroXTelemetry, MemoryReadableEmulator


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
        self._observation_adapter = ResizedObservationAdapter(config.observation)
        self._observation_stack = FrameStackBuffer(
            frame_space=self._observation_adapter.observation_space,
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
        frame, info = self._reset_race_state()
        telemetry = _read_live_telemetry(self.backend)
        self._reward_tracker.reset(telemetry)
        self._episode_limits.reset(telemetry)
        self._episode_done = False
        self._episode_return = 0.0
        info["seed"] = seed
        observation = self._reset_observation(frame, info)
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
            (
                latest_frame,
                step_reward,
                terminated,
                truncated,
                info,
                last_telemetry,
            ) = self._advance_one_frame()
            total_reward += step_reward
            info["repeat_index"] = repeat_index

            if terminated or truncated:
                break

        if latest_frame is None:
            raise RuntimeError("The emulator did not produce a frame during step()")

        if last_telemetry is not None:
            info["telemetry"] = last_telemetry.to_dict()
        self._episode_return += total_reward
        info["episode_return"] = self._episode_return
        _ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        observation = self._append_observation(latest_frame, info)

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
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        _ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        observation = self._append_observation(frame, info)
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
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object], FZeroXTelemetry | None]:
        frame_step = self.backend.step_frame()
        info = dict(frame_step.info)
        telemetry = _read_live_telemetry(self.backend)
        reward_step = self._reward_tracker.step(telemetry)
        reward = frame_step.reward + reward_step.reward
        terminated = frame_step.terminated or reward_step.terminated
        limit_step = self._episode_limits.step(telemetry)
        truncation_reason = limit_step.truncation_reason
        truncated = frame_step.truncated or (truncation_reason is not None)
        info["step_reward"] = reward_step.reward
        if reward_step.breakdown:
            info["reward_breakdown"] = dict(reward_step.breakdown)
        info["episode_step"] = limit_step.step_count
        info["stalled_steps"] = limit_step.stalled_steps
        if truncation_reason is not None:
            info["truncation_reason"] = truncation_reason
        if telemetry is not None:
            info.update(_telemetry_info(telemetry))
            termination_reason = _termination_reason(telemetry)
            if termination_reason is not None:
                info["termination_reason"] = termination_reason
        return (
            frame_step.frame,
            reward,
            terminated,
            truncated,
            info,
            telemetry,
        )

    def _reset_observation(self, frame: np.ndarray, info: dict[str, object]) -> np.ndarray:
        observation_frame = self._observation_adapter.transform(frame, info=info)
        observation = self._observation_stack.reset(observation_frame)
        info["observation_shape"] = tuple(int(value) for value in observation.shape)
        info["observation_frame_shape"] = self._observation_stack.frame_shape
        info["observation_stack"] = self.config.observation.frame_stack
        return observation

    def _append_observation(self, frame: np.ndarray, info: dict[str, object]) -> np.ndarray:
        observation_frame = self._observation_adapter.transform(frame, info=info)
        observation = self._observation_stack.append(observation_frame)
        info["observation_shape"] = tuple(int(value) for value in observation.shape)
        info["observation_frame_shape"] = self._observation_stack.frame_shape
        info["observation_stack"] = self.config.observation.frame_stack
        return observation

def _has_custom_baseline(info: dict[str, object]) -> bool:
    baseline_kind = info.get("baseline_kind")
    return baseline_kind == "custom"


def _read_live_telemetry(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    if not hasattr(backend, "system_ram_size") or not hasattr(backend, "read_system_ram"):
        return None
    try:
        return read_telemetry(cast(MemoryReadableEmulator, backend))
    except RuntimeError:
        return None


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
        "energy": telemetry.player.energy,
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


def _ensure_monitor_info_keys(info: dict[str, object]) -> None:
    info.setdefault("episode_return", 0.0)
    info.setdefault("episode_step", 0)
    info.setdefault("termination_reason", None)
    info.setdefault("truncation_reason", None)
    info.setdefault("race_distance", None)
    info.setdefault("speed_kph", None)
    info.setdefault("position", None)
    info.setdefault("lap", None)
