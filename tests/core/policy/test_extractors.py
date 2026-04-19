# tests/core/policy/test_extractors.py
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor


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


def test_observation_extractor_auto_features_dim_uses_medium_flatten() -> None:
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


def test_observation_extractor_accepts_large_preset() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = torch.zeros((2, 116, 164, 12), dtype=torch.float32)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_large_deep_flatten() -> None:
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


def test_observation_extractor_auto_features_dim_uses_compact_deep_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(98, 130, 9), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 98, 130, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 3_072
    assert tuple(features.shape) == (2, extractor._flatten_dim)


def test_observation_extractor_compact_deep_profile_can_override_small_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="compact_deep",
    )

    observations = torch.zeros((2, 84, 116, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_920
    assert tuple(features.shape) == (2, 1_920)


def test_observation_extractor_compact_bottleneck_profile_uses_compact_input() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(98, 130, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="compact_bottleneck",
    )

    observations = torch.zeros((2, 98, 130, 6), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 768
    assert tuple(features.shape) == (2, 768)


def test_observation_extractor_compact_deep_supports_tiny_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(66, 82, 6), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 66, 82, 6), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 768
    assert tuple(features.shape) == (2, 768)


def test_image_state_extractor_concatenates_cnn_and_state_features() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
            }
        ),
        features_dim=512,
        state_features_dim=64,
    )

    features = extractor(
        {
            "image": torch.zeros((2, 116, 164, 12), dtype=torch.float32),
            "state": torch.zeros((2, 11), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 576
    assert tuple(features.shape) == (2, 576)


def test_image_state_extractor_can_fuse_concatenated_features() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
            }
        ),
        features_dim=512,
        state_features_dim=64,
        fusion_features_dim=512,
    )

    features = extractor(
        {
            "image": torch.zeros((2, 116, 164, 12), dtype=torch.float32),
            "state": torch.zeros((2, 11), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 512
    assert tuple(features.shape) == (2, 512)
    assert isinstance(extractor._fusion_mlp, torch.nn.Sequential)


def test_image_state_extractor_auto_features_dim_uses_image_flatten_plus_state_branch() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
            }
        ),
        features_dim="auto",
        state_features_dim=64,
    )

    features = extractor(
        {
            "image": torch.zeros((2, 116, 164, 12), dtype=torch.float32),
            "state": torch.zeros((2, 11), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 3_648
    assert tuple(features.shape) == (2, 3_648)


def test_image_state_extractor_auto_features_dim_respects_conv_profile() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(98, 130, 9), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32),
            }
        ),
        features_dim="auto",
        state_features_dim=32,
        conv_profile="compact_deep",
    )

    features = extractor(
        {
            "image": torch.zeros((2, 98, 130, 9), dtype=torch.float32),
            "state": torch.zeros((2, 14), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 3_104
    assert tuple(features.shape) == (2, 3_104)
