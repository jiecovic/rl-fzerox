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


def test_observation_extractor_auto_features_dim_uses_large_nature_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 116, 164, 12), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 11_968
    assert tuple(features.shape) == (2, extractor._flatten_dim)


def test_observation_extractor_auto_features_dim_uses_wide_nature_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(98, 130, 9), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 98, 130, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor.features_dim == extractor._flatten_dim
    assert isinstance(extractor._linear, torch.nn.Identity)
    assert extractor._flatten_dim == 6_144
    assert tuple(features.shape) == (2, extractor._flatten_dim)


def test_observation_extractor_nature_profile_supports_wide_68x108_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(68, 108, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    observations = torch.zeros((2, 68, 108, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 3_200
    assert tuple(features.shape) == (2, 3_200)


def test_observation_extractor_nature_profile_supports_square_dqn_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 84, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    observations = torch.zeros((2, 84, 84, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 3_136
    assert tuple(features.shape) == (2, 3_136)


def test_observation_extractor_nature_profile_supports_compact_aspect_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(76, 100, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    observations = torch.zeros((2, 76, 100, 9), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 3_456
    assert tuple(features.shape) == (2, 3_456)


def test_observation_extractor_custom_profile_uses_configured_layers() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="custom",
        custom_conv_layers=(
            {"out_channels": 16, "kernel_size": 6, "stride": 3, "padding": 0},
            {"out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
            {"out_channels": 48, "kernel_size": 3, "stride": 1, "padding": 0},
        ),
    )

    observations = torch.zeros((2, 60, 76, 6), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 2_592
    assert tuple(features.shape) == (2, 2_592)
    assert [
        module.out_channels for module in extractor._cnn if isinstance(module, torch.nn.Conv2d)
    ] == [16, 32, 48]


def test_observation_extractor_custom_profile_supports_padding() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="custom",
        custom_conv_layers=(
            {"out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 24, "kernel_size": 3, "stride": 2, "padding": 1},
        ),
    )

    observations = torch.zeros((2, 60, 76, 6), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 6_840
    assert tuple(features.shape) == (2, 6_840)
    assert [module.padding for module in extractor._cnn if isinstance(module, torch.nn.Conv2d)] == [
        (1, 1),
        (1, 1),
    ]


def test_observation_extractor_rejects_custom_profile_that_collapses_geometry() -> None:
    with pytest.raises(ValueError, match="collapses"):
        FZeroXObservationCnnExtractor(
            spaces.Box(low=0, high=255, shape=(60, 76, 6), dtype=np.uint8),
            features_dim="auto",
            conv_profile="custom",
            custom_conv_layers=({"out_channels": 32, "kernel_size": 64, "stride": 1},),
        )


def test_observation_extractor_reports_convolution_activations() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 5), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    activations = extractor.convolution_activations(
        torch.zeros((1, 60, 76, 5), dtype=torch.float32)
    )

    assert [(name, tuple(values.shape)) for name, values in activations] == [
        ("conv1", (1, 32, 14, 18)),
        ("conv2", (1, 64, 6, 8)),
        ("conv3", (1, 64, 4, 6)),
    ]


def test_observation_extractor_nature_profile_supports_66x82_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(66, 82, 6), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 66, 82, 6), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_536
    assert tuple(features.shape) == (2, 1_536)


def test_observation_extractor_nature_profile_supports_clean_compact_aspect_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 5), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 60, 76, 5), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_536
    assert tuple(features.shape) == (2, 1_536)


def test_observation_extractor_nature_profile_supports_clean_square_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(68, 68, 5), dtype=np.uint8),
        features_dim="auto",
    )

    observations = torch.zeros((2, 68, 68, 5), dtype=torch.float32)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_600
    assert tuple(features.shape) == (2, 1_600)


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


def test_image_state_extractor_can_layer_norm_fused_features() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(66, 82, 6), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32),
            }
        ),
        features_dim="auto",
        state_features_dim=64,
        layer_norm=True,
    )

    features = extractor(
        {
            "image": torch.ones((2, 66, 82, 6), dtype=torch.float32),
            "state": torch.ones((2, 11), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 1_600
    assert tuple(features.shape) == (2, 1_600)
    assert isinstance(extractor._layer_norm, torch.nn.LayerNorm)


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

    assert extractor.features_dim == 12_032
    assert tuple(features.shape) == (2, 12_032)


def test_image_state_extractor_auto_features_dim_supports_68x108_geometry() -> None:
    extractor = FZeroXImageStateExtractor(
        spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(68, 108, 9), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32),
            }
        ),
        features_dim="auto",
        state_features_dim=32,
    )

    features = extractor(
        {
            "image": torch.zeros((2, 68, 108, 9), dtype=torch.float32),
            "state": torch.zeros((2, 14), dtype=torch.float32),
        }
    )

    assert extractor.features_dim == 3_232
    assert tuple(features.shape) == (2, 3_232)
