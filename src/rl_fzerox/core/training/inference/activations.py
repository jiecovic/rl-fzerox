# src/rl_fzerox/core/training/inference/activations.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeGuard

import numpy as np
import torch
from stable_baselines3.common.preprocessing import preprocess_obs

from fzerox_emulator.arrays import Float32Array


@dataclass(frozen=True, slots=True)
class PolicyCnnActivation:
    """One convolution layer activation map captured from policy inference."""

    name: str
    values: Float32Array


class _HasConvolutionActivations(Protocol):
    def convolution_activations(
        self,
        observations: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]: ...


def collect_policy_cnn_activations(
    policy: object,
    observation: object,
) -> tuple[PolicyCnnActivation, ...]:
    """Return CNN activation maps for the policy image extractor, if available."""

    policy_network = _policy_network(policy)
    image_extractor = _policy_image_extractor(policy_network)
    if image_extractor is None:
        return ()

    obs_to_tensor = getattr(policy_network, "obs_to_tensor", None)
    observation_space = getattr(policy_network, "observation_space", None)
    if not callable(obs_to_tensor) or observation_space is None:
        return ()

    tensor_result = obs_to_tensor(observation)
    if not isinstance(tensor_result, tuple) or len(tensor_result) != 2:
        return ()
    tensor_observation = tensor_result[0]
    normalize_images = bool(getattr(policy_network, "normalize_images", True))
    with torch.no_grad():
        preprocessed = preprocess_obs(
            tensor_observation,
            observation_space,
            normalize_images=normalize_images,
        )
        image_tensor = _image_tensor(preprocessed)
        activations = image_extractor.convolution_activations(image_tensor)

    return tuple(
        PolicyCnnActivation(
            name=name,
            values=np.ascontiguousarray(layer[0].detach().cpu().numpy(), dtype=np.float32),
        )
        for name, layer in activations
        if layer.ndim == 4 and layer.shape[0] > 0
    )


def _policy_network(policy: object) -> object:
    return getattr(policy, "policy", policy)


def _policy_image_extractor(policy_network: object) -> _HasConvolutionActivations | None:
    for candidate in _feature_extractor_candidates(policy_network):
        image_extractor = getattr(candidate, "_image_extractor", None)
        if _has_activation_api(image_extractor):
            return image_extractor
        if _has_activation_api(candidate):
            return candidate
    return None


def _feature_extractor_candidates(policy_network: object) -> tuple[object, ...]:
    candidates: list[object] = []
    for attr_name in ("pi_features_extractor", "features_extractor"):
        candidate = getattr(policy_network, attr_name, None)
        if candidate is not None:
            candidates.append(candidate)

    actor = getattr(policy_network, "actor", None)
    actor_extractor = None if actor is None else getattr(actor, "features_extractor", None)
    if actor_extractor is not None:
        candidates.append(actor_extractor)
    return tuple(candidates)


def _has_activation_api(candidate: object) -> TypeGuard[_HasConvolutionActivations]:
    return callable(getattr(candidate, "convolution_activations", None))


def _image_tensor(preprocessed_observation: object) -> torch.Tensor:
    if isinstance(preprocessed_observation, Mapping):
        image = preprocessed_observation.get("image")
        if not isinstance(image, torch.Tensor):
            raise TypeError("Preprocessed dict observation does not contain tensor key 'image'")
        return image
    if not isinstance(preprocessed_observation, torch.Tensor):
        raise TypeError("Preprocessed observation is not a tensor")
    return preprocessed_observation
