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
from rl_fzerox.core.runtime_spec.schema import PolicyConfig


def maybe_wrap_training_auxiliary_state_observation(
    env: gym.Env,
    *,
    policy_config: PolicyConfig,
) -> gym.Env:
    """Expose hidden aux targets in dict observations when the head bank is enabled."""

    if not policy_config.auxiliary_state.enabled:
        return env
    return AuxiliaryStateObservationWrapper(env)


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
        return raw_targets.astype(np.float32, copy=False)
    return auxiliary_state_targets_from_mapping(info)
