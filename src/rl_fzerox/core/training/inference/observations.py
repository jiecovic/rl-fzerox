# src/rl_fzerox/core/training/inference/observations.py
"""Observation preparation for policy inference outside the training env."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator.arrays import ObservationFrame
from rl_fzerox.core.envs.observations import ImageStateObservation, ObservationValue
from rl_fzerox.core.policy.auxiliary_state import auxiliary_state_target_spec
from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
    mapping_has_auxiliary_state_targets,
)


def observation_for_policy(policy: object, observation: ObservationValue) -> ObservationValue:
    observation_space = policy_observation_space(policy)
    if isinstance(observation_space, spaces.Box):
        return _adapt_box_observation_for_policy(observation, observation_space)
    if not isinstance(observation_space, spaces.Dict):
        return observation

    dict_space = observation_space
    observation = _adapt_dict_observation_for_policy(observation, dict_space)
    field_name = auxiliary_state_targets_field()
    dict_fields = getattr(dict_space, "spaces", {})
    if not isinstance(dict_fields, Mapping) or field_name not in dict_fields:
        return observation
    if not isinstance(observation, dict):
        raise TypeError("Auxiliary-state policy expects dict observations")
    if mapping_has_auxiliary_state_targets(observation):
        return observation

    zeros = np.zeros(auxiliary_state_target_spec().count, dtype=np.float32)
    augmented_observation: ImageStateObservation = {
        "image": observation["image"],
        "state": observation["state"],
        "auxiliary_state_targets": zeros,
    }
    return augmented_observation


def policy_observation_space(policy: object) -> spaces.Space | None:
    observation_space = getattr(policy, "observation_space", None)
    if isinstance(observation_space, spaces.Space):
        return observation_space
    inner_policy = getattr(policy, "policy", None)
    inner_observation_space = getattr(inner_policy, "observation_space", None)
    if isinstance(inner_observation_space, spaces.Space):
        return inner_observation_space
    return None


def _adapt_box_observation_for_policy(
    observation: ObservationValue,
    observation_space: spaces.Box,
) -> ObservationValue:
    if isinstance(observation, dict):
        return observation
    return _adapt_image_array_for_space(observation, observation_space)


def _adapt_dict_observation_for_policy(
    observation: ObservationValue,
    observation_space: spaces.Dict,
) -> ObservationValue:
    if not isinstance(observation, dict):
        return observation

    image_space = observation_space.spaces.get("image")
    if not isinstance(image_space, spaces.Box):
        return observation

    image = observation["image"]
    adapted_image = _adapt_image_array_for_space(image, image_space)
    if adapted_image is image:
        return observation
    adapted_observation: ImageStateObservation = {
        "image": adapted_image,
        "state": observation["state"],
    }
    auxiliary_state_targets = observation.get("auxiliary_state_targets")
    if isinstance(auxiliary_state_targets, np.ndarray):
        adapted_observation["auxiliary_state_targets"] = auxiliary_state_targets
    return adapted_observation


def _adapt_image_array_for_space(
    image: ObservationFrame,
    observation_space: spaces.Box,
) -> ObservationFrame:
    """Adapt channels-last image observations to channels-first policy spaces."""

    expected_shape = tuple(int(value) for value in observation_space.shape)
    if image.shape == expected_shape:
        return image
    if len(expected_shape) != 3 or image.ndim != 3:
        return image

    expected_channels, expected_height, expected_width = expected_shape
    if image.shape == (expected_height, expected_width, expected_channels):
        return np.transpose(image, (2, 0, 1))
    return image
