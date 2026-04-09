# src/rl_fzerox/core/envs/engine/runtime.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState, EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.boot import boot_into_first_race, continue_to_next_race
from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, RewardConfig
from rl_fzerox.core.envs.actions import BOOST_MASK, ActionValue, build_action_adapter
from rl_fzerox.core.envs.info import ensure_monitor_info_keys
from rl_fzerox.core.envs.observations import (
    ObservationValue,
    build_observation,
    build_observation_space,
)
from rl_fzerox.core.envs.rewards import RewardActionContext, build_reward_tracker
from rl_fzerox.core.seed import derive_seed

from .control_state import ControlStateTracker
from .info import (
    backend_step_info,
    has_custom_baseline,
    read_live_telemetry,
    reset_context_info,
    set_curriculum_info,
    set_observation_info,
    telemetry_can_boost,
    telemetry_info,
)
from .masks import ActionMaskController

_DOMAIN_RESET_RNG = 0xD6E8_2BC9_2A5F_1873


class FZeroXEnvEngine:
    """Environment step engine around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply reward shaping,
    build policy observations, and assemble Gym-facing info.
    """

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._action_adapter = build_action_adapter(config.action)
        self._observation_spec = backend.observation_spec(config.observation.preset)
        self._reward_tracker = build_reward_tracker(
            config=reward_config,
            max_episode_steps=config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._action_space = self._action_adapter.action_space
        self._observation_space = build_observation_space(
            self._observation_spec,
            frame_stack=config.observation.frame_stack,
            mode=config.observation.mode,
        )
        self._mask_controller = ActionMaskController.from_config(
            adapter=self._action_adapter,
            base_overrides=(
                config.action.mask.branch_overrides() if config.action.mask is not None else None
            ),
            curriculum_config=curriculum_config,
        )
        self._control_state = ControlStateTracker()
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self._last_info: dict[str, object] = {}
        self._reset_count = 0
        self._rng_seed_base: int | None = None

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def action_masks(self) -> np.ndarray:
        """Return the flattened boolean action mask for the current stage."""

        return self._mask_controller.action_mask()

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        self._mask_controller.set_curriculum_stage(stage_index)

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._mask_controller.stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        return self._mask_controller.stage_name

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode and return the first policy observation."""

        _, info, telemetry = self._reset_race_state()
        if seed is not None:
            self._rng_seed_base = seed
        telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
        self._mask_controller.set_boost_unlocked(telemetry_can_boost(telemetry))
        self._reward_tracker.reset(telemetry)
        self._episode_done = False
        self._episode_return = 0.0
        self._held_controller_state = ControllerState()
        self._control_state.reset()
        info["seed"] = seed
        set_curriculum_info(
            info,
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        image_observation = self._render_observation_image()
        observation = self._build_observation(image=image_observation, telemetry=telemetry)
        set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
        )
        self._last_info = dict(info)
        self._reset_count += 1
        return observation, info

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.step_control(self._action_adapter.decode(action))

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        """Decode one policy action into the held controller state it represents."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: ControllerState,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        self._held_controller_state = control_state
        return self._run_env_step(
            control_state,
            action_repeat=self.config.action_repeat,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        if control_state is not None:
            self._held_controller_state = control_state
        return self._run_env_step(self._held_controller_state, action_repeat=1)

    def render(self) -> np.ndarray:
        return self.backend.render_display(preset=self.config.observation.preset)

    def close(self) -> None:
        self.backend.close()

    def _reset_race_state(self) -> tuple[np.ndarray, dict[str, object], FZeroXTelemetry | None]:
        continue_error: str | None = None
        if self.config.reset_to_race and not has_custom_baseline(self._last_info):
            if self._episode_done:
                try:
                    frame, reset_info = continue_to_next_race(self.backend)
                    info = reset_context_info(self._last_info)
                    info.update(reset_info)
                    return frame, info, read_live_telemetry(self.backend)
                except RuntimeError as exc:
                    continue_error = str(exc)

        reset_state = self.backend.reset()
        info = dict(reset_state.info)
        frame = reset_state.frame
        if continue_error is not None:
            info["reset_fallback"] = "continue_to_next_race_failed"
            info["continue_to_next_race_error"] = continue_error

        if self.config.reset_to_race and not has_custom_baseline(info):
            frame, boot_info = boot_into_first_race(self.backend)
            info.update(boot_info)
            return frame, info, read_live_telemetry(self.backend)

        return frame, info, read_live_telemetry(self.backend)

    def _maybe_randomize_game_rng(
        self,
        seed: int | None,
        telemetry: FZeroXTelemetry | None,
        info: dict[str, object],
    ) -> FZeroXTelemetry | None:
        if not self.config.randomize_game_rng_on_reset:
            return telemetry
        seed_base = seed if seed is not None else self._rng_seed_base
        if seed_base is None:
            return telemetry
        if self.config.randomize_game_rng_requires_race_mode and (
            telemetry is None or not telemetry.in_race_mode
        ):
            info["rng_randomized"] = False
            info["rng_randomization_skip_reason"] = "not_in_race"
            return telemetry

        rng_seed = derive_seed(seed_base, _DOMAIN_RESET_RNG, self._reset_count)
        if rng_seed is None:
            return telemetry
        rng_state = self.backend.randomize_game_rng(rng_seed)
        info["rng_randomized"] = True
        info["rng_seed"] = rng_seed
        info["rng_state"] = rng_state
        return read_live_telemetry(self.backend) or telemetry

    def _run_env_step(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        step_result = self.backend.step_repeat_raw(
            control_state,
            action_repeat=action_repeat,
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
            stuck_min_speed_kph=float(self.config.stuck_min_speed_kph),
            energy_loss_epsilon=self._reward_summary_config.energy_loss_epsilon,
            max_episode_steps=self.config.max_episode_steps,
            stuck_step_limit=self.config.stuck_step_limit,
            wrong_way_timer_limit=self.config.wrong_way_timer_limit,
            terminate_on_energy_depleted=self.config.terminate_on_energy_depleted,
        )
        info = backend_step_info(self.backend)
        telemetry = step_result.telemetry
        self._mask_controller.set_boost_unlocked(telemetry_can_boost(telemetry))
        reward_step = self._reward_tracker.step_summary(
            step_result.summary,
            step_result.status,
            telemetry,
            RewardActionContext(
                boost_requested=bool(control_state.joypad_mask & BOOST_MASK),
            ),
        )
        reward = reward_step.reward
        reward_breakdown = dict(reward_step.breakdown)
        terminated = step_result.status.terminated
        truncated = step_result.status.truncated
        info["step_reward"] = reward
        info["repeat_index"] = max(step_result.summary.frames_run - 1, 0)
        if reward_breakdown:
            info["reward_breakdown"] = reward_breakdown
        info["episode_step"] = step_result.status.step_count
        info["stalled_steps"] = step_result.status.stalled_steps
        info["reverse_timer"] = step_result.status.reverse_timer
        info["termination_reason"] = step_result.status.termination_reason
        info["truncation_reason"] = step_result.status.truncation_reason
        set_curriculum_info(
            info,
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            # Keep env info pickle-safe for SubprocVecEnv workers.
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        self._control_state.record_step(
            control_state=control_state,
            frames_run=step_result.summary.frames_run,
        )
        image_observation = step_result.observation
        observation = self._build_observation(image=image_observation, telemetry=telemetry)
        self._episode_return += reward
        info["episode_return"] = self._episode_return
        ensure_monitor_info_keys(info)
        self._episode_done = terminated or truncated
        self._last_info = dict(info)
        set_observation_info(
            info,
            observation_shape=tuple(int(value) for value in image_observation.shape),
            observation_spec=self._observation_spec,
            frame_stack=self.config.observation.frame_stack,
            observation_mode=self.config.observation.mode,
        )
        return observation, reward, terminated, truncated, info

    def _render_observation_image(self) -> np.ndarray:
        return self.backend.render_observation(
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
        )

    def _build_observation(
        self,
        *,
        image: np.ndarray,
        telemetry: FZeroXTelemetry | None,
    ) -> ObservationValue:
        """Build the policy observation from the rendered image plus control context."""

        return build_observation(
            image=image,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            **self._control_state.observation_fields(),
        )
