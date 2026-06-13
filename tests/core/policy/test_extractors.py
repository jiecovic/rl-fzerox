# tests/core/policy/test_extractors.py
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor
from rl_fzerox.core.policy.extractors import PreActivationResidualConvBlock


def _zeros(*shape: int) -> torch.Tensor:
    return torch.Tensor(*shape).zero_()


def _ones(*shape: int) -> torch.Tensor:
    return torch.Tensor(*shape).fill_(1.0)


def test_observation_extractor_accepts_channels_last_observations() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 12), dtype=np.uint8),
        features_dim=512,
    )

    observations = _zeros(2, 84, 116, 12)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_accepts_channels_first_observations() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(12, 84, 116), dtype=np.uint8),
        features_dim=512,
    )

    observations = _zeros(2, 12, 84, 116)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_flatten_without_projection() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = _zeros(2, 84, 116, 12)
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

    observations = _zeros(2, 92, 124, 12)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_uses_configured_projection_activation() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 116, 12), dtype=np.uint8),
        features_dim=512,
        image_projection_activation="gelu",
    )

    assert isinstance(extractor._linear, torch.nn.Sequential)
    assert isinstance(extractor._linear[1], torch.nn.GELU)


def test_observation_extractor_auto_features_dim_uses_medium_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(92, 124, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = _zeros(2, 92, 124, 12)
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

    observations = _zeros(2, 116, 164, 12)
    features = extractor(observations)

    assert tuple(features.shape) == (2, 512)


def test_observation_extractor_auto_features_dim_uses_large_nature_flatten() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(116, 164, 12), dtype=np.uint8),
        features_dim="auto",
    )

    observations = _zeros(2, 116, 164, 12)
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

    observations = _zeros(2, 98, 130, 9)
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

    observations = _zeros(2, 68, 108, 9)
    features = extractor(observations)

    assert extractor._flatten_dim == 3_200
    assert tuple(features.shape) == (2, 3_200)


def test_observation_extractor_nature_profile_supports_square_dqn_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 84, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    observations = _zeros(2, 84, 84, 9)
    features = extractor(observations)

    assert extractor._flatten_dim == 3_136
    assert tuple(features.shape) == (2, 3_136)


def test_observation_extractor_impala_small_profile_uses_shallow_paper_stack() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(72, 96, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="impala_small",
    )

    observations = _zeros(2, 72, 96, 6)
    features = extractor(observations)
    activations = extractor.convolution_activations(observations[:1])

    assert extractor._flatten_dim == 2_240
    assert tuple(features.shape) == (2, 2_240)
    assert [(name, tuple(values.shape)) for name, values in activations] == [
        ("conv1", (1, 16, 17, 23)),
        ("conv2", (1, 32, 7, 10)),
    ]


def test_observation_extractor_impala_large_profile_uses_residual_conv_sequences() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(72, 96, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="impala_large",
    )

    observations = _zeros(2, 72, 96, 6)
    features = extractor(observations)
    activations = extractor.convolution_activations(observations[:1])

    assert extractor._flatten_dim == 3_456
    assert tuple(features.shape) == (2, 3_456)
    assert [(name, tuple(values.shape)) for name, values in activations] == [
        ("conv1", (1, 16, 72, 96)),
        ("pool2", (1, 16, 36, 48)),
        ("res-pre3", (1, 16, 36, 48)),
        ("res-pre4", (1, 16, 36, 48)),
        ("conv5", (1, 32, 36, 48)),
        ("pool6", (1, 32, 18, 24)),
        ("res-pre7", (1, 32, 18, 24)),
        ("res-pre8", (1, 32, 18, 24)),
        ("conv9", (1, 32, 18, 24)),
        ("pool10", (1, 32, 9, 12)),
        ("res-pre11", (1, 32, 9, 12)),
        ("res-pre12", (1, 32, 9, 12)),
        ("act13", (1, 32, 9, 12)),
    ]
    cnn_modules = tuple(extractor._cnn.children())
    assert isinstance(cnn_modules[0], torch.nn.Conv2d)
    assert isinstance(cnn_modules[1], torch.nn.MaxPool2d)
    assert isinstance(cnn_modules[2], PreActivationResidualConvBlock)
    assert isinstance(cnn_modules[-2], torch.nn.ReLU)
    assert isinstance(cnn_modules[-1], torch.nn.Flatten)


def test_observation_extractor_nature_profile_supports_compact_aspect_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(76, 100, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    observations = _zeros(2, 76, 100, 9)
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

    observations = _zeros(2, 60, 76, 6)
    features = extractor(observations)

    assert extractor._flatten_dim == 2_592
    assert tuple(features.shape) == (2, 2_592)
    assert [
        module.out_channels
        for module in extractor._cnn.modules()
        if isinstance(module, torch.nn.Conv2d)
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

    observations = _zeros(2, 60, 76, 6)
    features = extractor(observations)

    assert extractor._flatten_dim == 6_840
    assert tuple(features.shape) == (2, 6_840)
    assert [
        module.padding for module in extractor._cnn.modules() if isinstance(module, torch.nn.Conv2d)
    ] == [(1, 1), (1, 1)]


def test_observation_extractor_custom_profile_supports_residual_blocks() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="custom",
        custom_conv_layers=(
            {"kind": "conv", "out_channels": 16, "kernel_size": 6, "stride": 3, "padding": 0},
            {
                "kind": "residual_post",
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {"kind": "conv", "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 0},
        ),
    )

    observations = _zeros(2, 60, 76, 6)
    features = extractor(observations)
    activations = extractor.convolution_activations(observations[:1])

    assert extractor._flatten_dim == 2_816
    assert tuple(features.shape) == (2, 2_816)
    assert [type(module).__name__ for module in extractor._cnn] == [
        "Conv2d",
        "ReLU",
        "PostActivationResidualConvBlock",
        "Conv2d",
        "ReLU",
        "Flatten",
    ]
    assert [name for name, _ in activations] == ["conv1", "res2", "conv3"]


@pytest.mark.parametrize(
    ("pool_kind", "pool_module", "pool_name"),
    (("maxpool", "MaxPool2d", "pool2"), ("avgpool", "AvgPool2d", "avgpool2")),
)
def test_observation_extractor_custom_profile_supports_pooling_layers(
    pool_kind: str,
    pool_module: str,
    pool_name: str,
) -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 6), dtype=np.uint8),
        features_dim="auto",
        conv_profile="custom",
        custom_conv_layers=(
            {"kind": "conv", "out_channels": 16, "kernel_size": 6, "stride": 3, "padding": 0},
            {"kind": pool_kind, "kernel_size": 2, "stride": 2, "padding": 0},
            {"kind": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0},
        ),
    )

    observations = _zeros(2, 60, 76, 6)
    features = extractor(observations)
    activations = extractor.convolution_activations(observations[:1])

    assert extractor._flatten_dim == 2_240
    assert tuple(features.shape) == (2, 2_240)
    assert [type(module).__name__ for module in extractor._cnn] == [
        "Conv2d",
        "ReLU",
        pool_module,
        "Conv2d",
        "ReLU",
        "Flatten",
    ]
    assert [(name, tuple(values.shape)) for name, values in activations] == [
        ("conv1", (1, 16, 19, 24)),
        (pool_name, (1, 16, 9, 12)),
        ("conv3", (1, 32, 7, 10)),
    ]


def test_observation_extractor_conv_only_state_dict_keys_stay_legacy() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(84, 84, 9), dtype=np.uint8),
        features_dim="auto",
        conv_profile="nature",
    )

    state_keys = set(extractor.state_dict())

    assert "_cnn.0.weight" in state_keys
    assert "_cnn.2.weight" in state_keys
    assert "_cnn.4.weight" in state_keys
    assert "_cnn.0.conv.weight" not in state_keys


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

    activations = extractor.convolution_activations(_zeros(1, 60, 76, 5))

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

    observations = _zeros(2, 66, 82, 6)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_536
    assert tuple(features.shape) == (2, 1_536)


def test_observation_extractor_nature_profile_supports_clean_compact_aspect_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(60, 76, 5), dtype=np.uint8),
        features_dim="auto",
    )

    observations = _zeros(2, 60, 76, 5)
    features = extractor(observations)

    assert extractor._flatten_dim == 1_536
    assert tuple(features.shape) == (2, 1_536)


def test_observation_extractor_nature_profile_supports_clean_square_geometry() -> None:
    extractor = FZeroXObservationCnnExtractor(
        spaces.Box(low=0, high=255, shape=(68, 68, 5), dtype=np.uint8),
        features_dim="auto",
    )

    observations = _zeros(2, 68, 68, 5)
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
            "image": _zeros(2, 116, 164, 12),
            "state": _zeros(2, 11),
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
        state_activation="gelu",
        fusion_features_dim=512,
        fusion_activation="tanh",
    )

    features = extractor(
        {
            "image": _zeros(2, 116, 164, 12),
            "state": _zeros(2, 11),
        }
    )

    assert extractor.features_dim == 512
    assert tuple(features.shape) == (2, 512)
    assert isinstance(extractor._state_mlp, torch.nn.Sequential)
    assert isinstance(extractor._state_mlp[1], torch.nn.GELU)
    assert isinstance(extractor._fusion_mlp, torch.nn.Sequential)
    assert isinstance(extractor._fusion_mlp[1], torch.nn.Tanh)


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
        layer_norm_activation="tanh",
    )

    features = extractor(
        {
            "image": _ones(2, 66, 82, 6),
            "state": _ones(2, 11),
        }
    )

    assert extractor.features_dim == 1_600
    assert tuple(features.shape) == (2, 1_600)
    assert isinstance(extractor._layer_norm, torch.nn.LayerNorm)
    assert isinstance(extractor._layer_norm_activation, torch.nn.Tanh)


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
            "image": _zeros(2, 116, 164, 12),
            "state": _zeros(2, 11),
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
            "image": _zeros(2, 68, 108, 9),
            "state": _zeros(2, 14),
        }
    )

    assert extractor.features_dim == 3_232
    assert tuple(features.shape) == (2, 3_232)
