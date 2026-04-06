# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.boot import boot_into_first_race
from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend


class FZeroXEnvEngine:
    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
    ) -> None:
        self.backend = backend
        self.config = config
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
        reset_state = self.backend.reset()
        info = dict(reset_state.info)
        frame = reset_state.frame

        if self.config.reset_to_race and not _has_saved_baseline(info):
            frame, boot_info = boot_into_first_race(self.backend)
            info.update(boot_info)

        info["seed"] = seed
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

        for repeat_index in range(self.config.action_repeat):
            frame_step = self.backend.step_frame()
            latest_frame = frame_step.frame
            total_reward += frame_step.reward
            terminated = frame_step.terminated
            truncated = frame_step.truncated
            info = dict(frame_step.info)
            info["repeat_index"] = repeat_index

            if terminated or truncated:
                break

        if latest_frame is None:
            raise RuntimeError("The emulator did not produce a frame during step()")

        return (
            np.array(latest_frame, copy=True),
            total_reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray:
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()


def _has_saved_baseline(info: dict[str, object]) -> bool:
    baseline_state_path = info.get("baseline_state_path")
    if not isinstance(baseline_state_path, str):
        return False
    return Path(baseline_state_path).is_file()
