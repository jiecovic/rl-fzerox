# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

from typing import cast

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
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
        self._reward_tracker = RewardTracker()
        self._episode_done = False
        self._last_info: dict[str, object] = {}
        self._action_space = spaces.Discrete(1)
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.backend.frame_shape,
            dtype=np.uint8,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        frame, info = self._reset_race_state()
        telemetry = _read_live_telemetry(self.backend)
        self._reward_tracker.reset(telemetry)
        self._episode_done = False
        info["seed"] = seed
        self._last_info = dict(info)
        return np.array(frame, copy=True), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if int(action) != 0:
            raise ValueError(f"Only action 0 is supported right now, got {action!r}")

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
        self._episode_done = terminated or truncated
        self._last_info = dict(info)

        return (
            np.array(latest_frame, copy=True),
            total_reward,
            terminated,
            truncated,
            info,
        )

    def step_frame(self) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        frame, reward, terminated, truncated, info, _ = self._advance_one_frame()
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        return np.array(frame, copy=True), reward, terminated, truncated, info

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
        truncated = frame_step.truncated
        info["step_reward"] = reward_step.reward
        if reward_step.breakdown:
            info["reward_breakdown"] = dict(reward_step.breakdown)
        return (
            frame_step.frame,
            reward,
            terminated,
            truncated,
            info,
            telemetry,
        )


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
