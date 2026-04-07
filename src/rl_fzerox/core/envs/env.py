# src/rl_fzerox/core/envs/env.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine import FZeroXEnvEngine


class FZeroXEnv(gym.Env[np.ndarray, np.ndarray]):
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
        return self._engine.reset(seed=seed)

    def step(self, action: ActionValue):
        return self._engine.step(action)

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        return self._engine.action_to_control_state(action)

    def step_control(self, control_state: ControllerState):
        return self._engine.step_control(control_state)

    def step_frame(self, control_state: ControllerState | None = None):
        return self._engine.step_frame(control_state)

    def render(self) -> np.ndarray:
        return self._engine.render()

    def close(self) -> None:
        self._engine.close()
