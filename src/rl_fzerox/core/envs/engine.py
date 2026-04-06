# src/rl_fzerox/core/envs/engine.py
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Space

from rl_fzerox.core.config.models import EnvConfig
from rl_fzerox.core.emulator.base import EmulatorBackend
from rl_fzerox.core.envs.contracts import ActionSpec, ObservationSpec, RewardFunction
from rl_fzerox.core.envs.defaults import BackendReward, NoOpActionSpec, RawFrameObservationSpec


class FZeroXEnvEngine:
    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        action_spec: ActionSpec | None = None,
        observation_spec: ObservationSpec | None = None,
        reward_function: RewardFunction | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.action_spec = action_spec or NoOpActionSpec()
        self.observation_spec = observation_spec or RawFrameObservationSpec()
        self.reward_function = reward_function or BackendReward()

    @property
    def action_space(self) -> Space[np.int64]:
        return self.action_spec.action_space

    @property
    def observation_space(self) -> Space[np.ndarray]:
        return self.observation_spec.observation_space(self.backend.frame_shape)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        reset_state = self.backend.reset(seed=seed)
        self.reward_function.reset()
        return self.observation_spec.transform(reset_state.frame), dict(reset_state.info)

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.action_spec.validate(action)

        latest_frame: np.ndarray | None = None
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, object] = {}

        for repeat_index in range(self.config.action_repeat):
            frame_step = self.backend.step_frame()
            latest_frame = frame_step.frame
            total_reward += self.reward_function.reward(frame_step)
            terminated = frame_step.terminated
            truncated = frame_step.truncated
            info = dict(frame_step.info)
            info["repeat_index"] = repeat_index

            if terminated or truncated:
                break

        if latest_frame is None:
            raise RuntimeError("The emulator did not produce a frame during step()")

        return (
            self.observation_spec.transform(latest_frame),
            total_reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray:
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()
