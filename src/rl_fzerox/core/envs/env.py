# src/rl_fzerox/core/envs/env.py
"""Public Gymnasium environment wrapper.

This file exposes `FZeroXEnv`, the stable object callers instantiate. Reset,
step, observation, reward, and watch behavior are delegated to runtime and
engine modules so this wrapper stays focused on the Gym API boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym

from fzerox_emulator import EmulatorBackend, RaceControlState, SpinRequest
from fzerox_emulator.arrays import ActionMask, RgbFrame, StateVector
from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.envs.actions import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.envs.gym_runtime import FZeroXEnvRuntime
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)


class FZeroXEnv(gym.Env[ObservationValue, ActionValue]):
    """Gymnasium wrapper around the emulator-backed F-Zero X environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        env_index: int = 0,
    ) -> None:
        super().__init__()
        self._runtime = FZeroXEnvRuntime(
            backend=backend,
            config=config,
            reward_config=reward_config,
            env_index=env_index,
        )
        self.backend = self._runtime.backend
        self.config = self._runtime.config
        self.action_space = self._runtime.action_space
        self.observation_space = self._runtime.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        """Reset the env and return the first stacked observation.

        The `seed` is accepted to match the Gym API and to seed any future
        Python-side reset randomization. The emulator reset path itself remains
        deterministic today.
        """

        _ = options
        super().reset(seed=seed)
        return self._runtime.reset(seed=seed)

    def step(self, action: ActionValue):
        return self._runtime.step(action)

    def step_watch(self, action: ActionValue):
        """Step a policy action and include watch-only intermediate display frames."""

        return self._runtime.step_watch(action)

    def action_to_control_state(self, action: ActionValue) -> RaceControlState:
        return self._runtime.action_to_control_state(action)

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._runtime.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._runtime.last_gas_level

    def action_masks(self) -> ActionMask:
        """Return the flattened boolean action mask for maskable PPO."""

        return self._runtime.action_masks()

    def action_mask_branches(self) -> ActionMaskBranches:
        """Return the mask grouped by action branch for watch/debug UI."""

        return self._runtime.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        """Return the flat mask and branch view from one mask computation."""

        return self._runtime.action_mask_snapshot()

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return ordered discrete action branches for policy-head initialization."""

        return self._runtime.action_dimensions

    def set_track_sampling_weights(self, weights_by_track_id: dict[str, float]) -> None:
        """Update adaptive reset weights used by step-balanced track sampling."""

        self._runtime.set_track_sampling_weights(weights_by_track_id)

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        """Replace the active reset candidates after generated-course rotation."""

        self._runtime.set_track_sampling_config(config)

    def set_engine_tuning_sampler(self, sampler: EngineTuningResetSampler | None) -> None:
        """Replace adaptive engine choices used at future resets."""

        self._runtime.set_engine_tuning_sampler(sampler)

    def set_engine_tuning_selection(self, selection: EngineTuningSelectionMode) -> None:
        """Choose whether adaptive engine tuning samples or picks greedy values."""

        self._runtime.set_engine_tuning_selection(selection)

    def extend_track_sampling_reset_queue(
        self,
        queued_resets: Sequence[TrackSamplingQueuedReset | str],
    ) -> None:
        """Append externally scheduled reset slots for deficit-budget sampling."""

        self._runtime.extend_track_sampling_reset_queue(queued_resets)

    def clear_track_sampling_reset_queue(self) -> None:
        """Drop unconsumed deficit-budget reset slots after scheduler replanning."""

        self._runtime.clear_track_sampling_reset_queue()

    def track_sampling_reset_queue_length(self) -> int:
        """Return queued deficit-budget reset course count."""

        return self._runtime.track_sampling_reset_queue_length()

    def set_locked_reset_course(self, course_id: str | None) -> None:
        """Lock subsequent sampled resets to one course for watch/manual inspection."""

        self._runtime.set_locked_reset_course(course_id)

    def set_sequential_track_sampling(self, enabled: bool) -> None:
        """Use configured track order for watch resets instead of training sampling."""

        self._runtime.set_sequential_track_sampling(enabled)

    def set_next_sequential_reset_course(self, course_id: str | None) -> None:
        """Align the next sequential watch reset to a specific configured course."""

        self._runtime.set_next_sequential_reset_course(course_id)

    def preload_track_baseline_paths(self, paths: Sequence[Path]) -> None:
        """Warm selected sampled-track savestates before watch navigation."""

        self._runtime.preload_track_baseline_paths(paths)

    def auxiliary_state_targets(self) -> StateVector:
        """Return the current hidden auxiliary-state target vector."""

        return self._runtime.auxiliary_state_targets()

    def step_control(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ):
        return self._runtime.step_control(control_state, spin_request=spin_request)

    def step_control_watch(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ):
        """Step manual controls and include watch-only intermediate display frames."""

        return self._runtime.step_control_watch(control_state, spin_request=spin_request)

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
        *,
        spin_request: SpinRequest = "none",
    ):
        return self._runtime.step_frame(control_state, spin_request=spin_request)

    def render(self) -> RgbFrame:
        return self._runtime.render()

    def close(self) -> None:
        self._runtime.close()
