# src/rl_fzerox/core/envs/gym_runtime/runtime.py
"""Stateful runtime backing the public Gym environment.

`FZeroXEnvRuntime` owns Gym-facing reset/step/render calls and delegates
low-level work to reset, stepping, action, and engine component modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from gymnasium import spaces

from fzerox_emulator import EmulatorBackend, RaceControlState, SpinRequest
from fzerox_emulator.arrays import ActionMask, RgbFrame, StateVector
from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.envs.actions import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.engine.components import build_engine_runtime_components
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    ActionMaskSnapshot,
)
from rl_fzerox.core.envs.engine.reset import EngineResetCoordinator
from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.envs.engine.stepping import WatchEnvStep
from rl_fzerox.core.envs.gym_runtime.reset import reset_gym_episode
from rl_fzerox.core.envs.gym_runtime.stepping import GymStepRuntime
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_vector_or_zeros,
)
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)


class FZeroXEnvRuntime:
    """Training/watch runtime around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply Gym termination,
    reward shaping, policy observations, and watch-facing info.
    """

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        env_index: int = 0,
    ) -> None:
        self.backend = backend
        self.config = config
        self._components = build_engine_runtime_components(
            backend=backend,
            config=config,
            reward_config=reward_config,
        )
        components = self._components
        self._renderer: RendererName = components.renderer
        self._mask_controller = components.mask_controller
        self._reset_coordinator = EngineResetCoordinator(
            backend=backend,
            config=config,
            env_index=env_index,
        )
        self._step_runtime = GymStepRuntime(config=config, components=components)
        self._episode = components.episode

    @property
    def action_space(self) -> spaces.Space:
        return self._step_runtime.action_space

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        return self._step_runtime.action_dimensions

    @property
    def observation_space(self) -> spaces.Space:
        return self._components.observation_builder.space

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._episode.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._episode.last_gas_level

    def action_masks(self) -> ActionMask:
        """Return the flattened boolean action mask for the current stage."""

        return self._mask_controller.action_mask()

    def action_mask_branches(self) -> ActionMaskBranches:
        """Return the current action mask grouped by branch label."""

        return self._mask_controller.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        """Return the flat mask and branch view from one mask computation."""

        return self._mask_controller.action_mask_snapshot()

    def set_track_sampling_weights(self, weights_by_track_id: dict[str, float]) -> None:
        """Update adaptive reset weights used by step-balanced track sampling."""

        self._reset_coordinator.set_track_sampling_weights(weights_by_track_id)

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        """Replace reset candidates after generated-course rotation."""

        self._reset_coordinator.set_track_sampling_config(config)

    def set_engine_tuning_sampler(self, sampler: EngineTuningResetSampler | None) -> None:
        """Replace adaptive engine choices used at future resets."""

        self._reset_coordinator.set_engine_tuning_sampler(sampler)

    def set_engine_tuning_selection(self, selection: EngineTuningSelectionMode) -> None:
        """Choose whether adaptive engine tuning samples or picks greedy values."""

        self._reset_coordinator.set_engine_tuning_selection(selection)

    def extend_track_sampling_reset_queue(
        self,
        queued_resets: Sequence[TrackSamplingQueuedReset | str],
    ) -> None:
        """Append externally scheduled reset slots for deficit-budget sampling."""

        self._reset_coordinator.extend_track_sampling_reset_queue(queued_resets)

    def clear_track_sampling_reset_queue(self) -> None:
        """Drop unconsumed deficit-budget reset slots after scheduler replanning."""

        self._reset_coordinator.clear_track_sampling_reset_queue()

    def track_sampling_reset_queue_length(self) -> int:
        """Return queued deficit-budget reset course count."""

        return self._reset_coordinator.track_sampling_reset_queue_length()

    def set_locked_reset_course(self, course_id: str | None) -> None:
        """Lock subsequent sampled resets to one course for watch/manual inspection."""

        self._reset_coordinator.set_locked_reset_course(course_id)

    def set_sequential_track_sampling(self, enabled: bool) -> None:
        """Use configured track order for watch resets instead of training sampling."""

        self._reset_coordinator.set_sequential_track_sampling(enabled)

    def set_next_sequential_reset_course(self, course_id: str | None) -> None:
        """Align the next sequential watch reset to a specific configured course."""

        self._reset_coordinator.set_next_sequential_course(course_id)

    def preload_track_baseline_paths(self, paths: Sequence[Path]) -> None:
        """Warm selected sampled-track savestates before manual watch navigation."""

        self._reset_coordinator.preload_track_baseline_paths(paths)

    def auxiliary_state_targets(self) -> StateVector:
        """Return the current hidden auxiliary-state target vector."""

        return auxiliary_state_target_vector_or_zeros(self._episode.last_telemetry)

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode and return the first policy observation."""

        return reset_gym_episode(
            backend=self.backend,
            config=self.config,
            components=self._components,
            reset_coordinator=self._reset_coordinator,
            seed=seed,
        )

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_runtime.step(action)

    def step_watch(self, action: ActionValue) -> WatchEnvStep:
        """Step a policy action while collecting watch-only intermediate frames."""

        return self._step_runtime.step_watch(action)

    def action_to_control_state(self, action: ActionValue) -> RaceControlState:
        """Decode one policy action into the held semantic control state."""

        return self._step_runtime.action_to_control_state(action)

    def step_control(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_runtime.step_control(control_state, spin_request=spin_request)

    def step_control_watch(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ) -> WatchEnvStep:
        """Step manual controls while collecting watch-only intermediate frames."""

        return self._step_runtime.step_control_watch(control_state, spin_request=spin_request)

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
        *,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        return self._step_runtime.step_frame(control_state, spin_request=spin_request)

    def render(self) -> RgbFrame:
        return self.backend.render_display(
            **self.config.observation.native_resolution_kwargs(renderer=self._renderer)
        )

    def close(self) -> None:
        self.backend.close()
