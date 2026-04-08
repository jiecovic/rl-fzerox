# tests/core/policy/test_extractors.py
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from rl_fzerox.core.policy import FZeroXObservationCnnExtractor


def test_observation_extractor_accepts_channels_last_observations() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 84, 116, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_accepts_channels_first_observations() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(12, 84, 116), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 12, 84, 116), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_flatten_without_projection() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 84, 116, 12), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 4_928
    assert tuple(features.shape) == (2, extractor._flatten_dim)


def test_observation_extractor_accepts_larger_aspect_correct_preset() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(92, 124, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 92, 124, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_richer_v2_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(92, 124, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 92, 124, 12), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 6_144
    assert tuple(features.shape) == (2, extractor._flatten_dim)


def test_observation_extractor_accepts_larger_default_v3_preset() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 116, 164, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_v3_legacy_deep_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 116, 164, 12), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 3_584
    assert tuple(features.shape) == (2, extractor._flatten_dim)
