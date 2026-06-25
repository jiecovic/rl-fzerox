# src/rl_fzerox/core/training/session/auxiliary_state.py
"""Auxiliary state-prediction wiring for training policy configuration."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
    auxiliary_state_targets_from_mapping,
    mapping_with_auxiliary_state_targets,
)
from rl_fzerox.core.policy.auxiliary_state.targets import auxiliary_state_target_vector_space
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig


def maybe_wrap_training_auxiliary_state_observation(
    env: gym.Env,
    *,
    policy_config: PolicyConfig,
    train_config: TrainConfig | None = None,
) -> gym.Env:
    """Expose hidden aux targets for policy-side losses that need state labels."""

    resolved_train_config = train_config or TrainConfig()
    if not _requires_auxiliary_state_targets(policy_config, resolved_train_config):
        return env
    return AuxiliaryStateObservationWrapper(env)


def _requires_auxiliary_state_targets(
    policy_config: PolicyConfig,
    train_config: TrainConfig,
) -> bool:
    return (
        policy_config.auxiliary_state.enabled
        or train_config.actor_regularization.requires_auxiliary_targets()
    )


class AuxiliaryStateObservationWrapper(gym.Wrapper):
    """Attach hidden supervised aux targets without changing actor-visible state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise RuntimeError(
                "policy auxiliary state requires dict observations; "
                "use observation.mode=image_state"
            )
        spaces_by_key = dict(env.observation_space.spaces)
        spaces_by_key[auxiliary_state_targets_field()] = auxiliary_state_target_vector_space()
        self.observation_space = spaces.Dict(spaces_by_key)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment_observation(observation, info), info

    def step(self, action: object):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_observation(observation, info), reward, terminated, truncated, info

    def _augment_observation(
        self,
        observation: object,
        info: dict[str, object],
    ) -> object:
        if not isinstance(observation, dict):
            raise TypeError("Auxiliary state wrapper requires dict observations")
        raw_targets = _current_auxiliary_state_targets(self.env, info)
        if raw_targets is None:
            raise KeyError(
                "Auxiliary state wrapper expected current auxiliary-state targets; "
                "training env plumbing dropped the hidden supervision target source."
            )
        targets: StateVector = raw_targets
        return mapping_with_auxiliary_state_targets(observation, targets=targets)


def _current_auxiliary_state_targets(
    env: gym.Env,
    info: dict[str, object],
) -> StateVector | None:
    env_targets = getattr(env, "auxiliary_state_targets", None)
    if callable(env_targets):
        raw_targets = env_targets()
        if not isinstance(raw_targets, np.ndarray):
            return None
        targets: StateVector = np.asarray(raw_targets, dtype=np.float32)
        return targets
    return auxiliary_state_targets_from_mapping(info)
