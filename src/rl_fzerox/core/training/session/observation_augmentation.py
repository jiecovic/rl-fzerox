# src/rl_fzerox/core/training/session/observation_augmentation.py
from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.config.schema import EnvConfig, TrainConfig
from rl_fzerox.core.envs.observations import (
    action_history_settings_for_observation,
    state_feature_names,
)

_COURSE_CONTEXT_PREFIX = "course_context."
_COURSE_CONTEXT_DROPOUT_SEED_DOMAIN = 0x34C2_2E5B_7B67_021D


def maybe_wrap_training_observation_augmentation(
    env: gym.Env,
    *,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> gym.Env:
    """Apply train-only observation augmentations without changing env spaces."""

    dropout_prob = float(train_config.course_context_dropout_prob)
    if dropout_prob <= 0.0:
        return env

    feature_indices = _course_context_feature_indices(env_config)
    if not feature_indices:
        return env

    return CourseContextDropoutWrapper(
        env,
        dropout_prob=dropout_prob,
        feature_indices=feature_indices,
    )


class CourseContextDropoutWrapper(gym.ObservationWrapper):
    """Randomly zero course-context state features for whole training episodes."""

    def __init__(
        self,
        env: gym.Env,
        *,
        dropout_prob: float,
        feature_indices: Sequence[int],
    ) -> None:
        super().__init__(env)
        self._dropout_prob = dropout_prob
        self._feature_indices = tuple(feature_indices)
        self._rng = np.random.default_rng()
        self._drop_current_episode = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed ^ _COURSE_CONTEXT_DROPOUT_SEED_DOMAIN)
        self._drop_current_episode = bool(self._rng.random() < self._dropout_prob)
        return self.observation(observation), info

    def observation(self, observation: object) -> object:
        if not self._drop_current_episode:
            return observation
        if not isinstance(observation, dict):
            return observation

        state = observation.get("state")
        if not isinstance(state, np.ndarray):
            return observation

        augmented_observation = dict(observation)
        augmented_state: StateVector = np.array(state, copy=True)
        augmented_state[list(self._feature_indices)] = 0.0
        augmented_observation["state"] = augmented_state
        return augmented_observation


def _course_context_feature_indices(config: EnvConfig) -> tuple[int, ...]:
    observation_config = config.observation
    state_components = observation_config.state_components_data()
    action_history_len, action_history_controls = action_history_settings_for_observation(
        state_components=state_components,
        fallback_len=observation_config.action_history_len,
        fallback_controls=observation_config.action_history_controls,
    )
    names = state_feature_names(
        observation_config.state_profile,
        course_context=observation_config.course_context,
        ground_effect_context=observation_config.ground_effect_context,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
        state_components=state_components,
    )
    return tuple(
        index for index, name in enumerate(names) if name.startswith(_COURSE_CONTEXT_PREFIX)
    )
