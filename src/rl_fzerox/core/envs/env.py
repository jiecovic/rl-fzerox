# src/rl_fzerox/core/envs/env.py
from __future__ import annotations

import gymnasium as gym

from fzerox_emulator import ControllerState, EmulatorBackend
from fzerox_emulator.arrays import ActionMask, RgbFrame
from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, RewardConfig
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine import FZeroXEnvEngine
from rl_fzerox.core.envs.observations import ObservationValue


class FZeroXEnv(gym.Env[ObservationValue, ActionValue]):
    """Gymnasium wrapper around the emulator-backed F-Zero X environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
    ) -> None:
        super().__init__()
        self._engine = FZeroXEnvEngine(
            backend=backend,
            config=config,
            reward_config=reward_config,
            curriculum_config=curriculum_config,
        )
        self.backend = self._engine.backend
        self.config = self._engine.config
        self.action_space = self._engine.action_space
        self.observation_space = self._engine.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        """Reset the env and return the first stacked observation.

        The `seed` is accepted to match the Gym API and to seed any future
        Python-side reset randomization. The emulator reset path itself remains
        deterministic today.
        """

        _ = options
        super().reset(seed=seed)
        return self._engine.reset(seed=seed)

    def step(self, action: ActionValue):
        return self._engine.step(action)

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        return self._engine.action_to_control_state(action)

    def action_masks(self) -> ActionMask:
        """Return the flattened boolean action mask for maskable PPO."""

        return self._engine.action_masks()

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage used for action masking."""

        self._engine.set_curriculum_stage(stage_index)

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        """Align watch-time curriculum masks with saved checkpoint metadata."""

        self._engine.sync_checkpoint_curriculum_stage(stage_index)

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._engine.curriculum_stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        return self._engine.curriculum_stage_name

    def step_control(self, control_state: ControllerState):
        return self._engine.step_control(control_state)

    def step_frame(self, control_state: ControllerState | None = None):
        return self._engine.step_frame(control_state)

    def render(self) -> RgbFrame:
        return self._engine.render()

    def close(self) -> None:
        self._engine.close()
