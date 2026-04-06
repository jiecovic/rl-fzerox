# src/rl_fzerox/core/envs/env.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.envs.engine import FZeroXEnvEngine


class FZeroXEnv(gym.Env[np.ndarray, np.int64]):
    """Gymnasium wrapper around the current raw-frame emulator backend."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        backend: EmulatorBackend,
        config: EnvConfig,
    ) -> None:
        super().__init__()
        self._engine = FZeroXEnvEngine(
            backend=backend,
            config=config,
        )
        self.backend = self._engine.backend
        self.config = self._engine.config
        self.action_space = self._engine.action_space
        self.observation_space = self._engine.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        _ = options
        super().reset(seed=seed)
        observation, info = self._engine.reset()
        info["seed"] = seed
        return observation, info

    def step(self, action: int | np.integer):
        return self._engine.step(action)

    def render(self) -> np.ndarray:
        return self._engine.render()

    def close(self) -> None:
        self._engine.close()
