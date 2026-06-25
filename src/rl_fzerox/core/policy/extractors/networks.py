# src/rl_fzerox/core/policy/extractors/networks.py
"""CNN and image-state feature extractors for policy observations."""

from __future__ import annotations

from typing import Literal

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rl_fzerox.core.policy.activations import ActivationName, resolve_policy_activation_fn
from rl_fzerox.core.policy.extractors.blocks import (
    PostActivationResidualConvBlock,
    PreActivationResidualConvBlock,
    torch_pooling_layer,
)
from rl_fzerox.core.policy.extractors.specs import (
    ConvProfile,
    ConvSpec,
    CustomConvLayerConfig,
    cnn_layer_name,
    cnn_layer_output_channels,
    image_flatten_dim,
    resolve_supported_image_geometry,
)


class FZeroXObservationCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for supported F-Zero X observation image geometries."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | Literal["auto"] = 512,
        conv_profile: ConvProfile = "nature",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
        image_projection_activation: ActivationName = "relu",
        layer_norm: bool = False,
        layer_norm_activation: ActivationName | None = None,
    ) -> None:
        image_geometry = resolve_supported_image_geometry(
            observation_space,
            extractor_name=type(self).__name__,
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
        )
        self._height = image_geometry.height
        self._width = image_geometry.width
        self._channels = image_geometry.channels
        self._conv_spec = image_geometry.conv_spec

        cnn_layers, capture_indices = self._build_conv_layers(
            input_channels=self._channels,
            conv_spec=self._conv_spec,
        )
        self._cnn_capture_indices = capture_indices
        cnn = nn.Sequential(*cnn_layers, nn.Flatten())

        with torch.no_grad():
            sample = torch.Tensor(1, self._channels, self._height, self._width).zero_()
            n_flatten = int(cnn(sample).shape[1])

        resolved_features_dim = n_flatten if features_dim == "auto" else int(features_dim)
        super().__init__(observation_space, resolved_features_dim)
        self._flatten_dim = n_flatten
        self._cnn = cnn
        if features_dim == "auto":
            self._linear: nn.Module = nn.Identity()
        else:
            self._linear = nn.Sequential(
                nn.Linear(n_flatten, resolved_features_dim),
                resolve_policy_activation_fn(image_projection_activation)(),
            )
        self._layer_norm: nn.Module = (
            nn.LayerNorm(resolved_features_dim) if layer_norm else nn.Identity()
        )
        self._layer_norm_activation = _layer_norm_activation_layer(
            layer_norm=layer_norm,
            activation=layer_norm_activation,
        )

    def _build_conv_layers(
        self,
        *,
        input_channels: int,
        conv_spec: ConvSpec,
    ) -> tuple[list[nn.Module], tuple[int, ...]]:
        layers: list[nn.Module] = []
        capture_indices: list[int] = []
        in_channels = input_channels
        for layer_spec in conv_spec:
            if layer_spec.kind == "conv":
                conv = nn.Conv2d(
                    in_channels,
                    layer_spec.out_channels,
                    kernel_size=layer_spec.kernel_size,
                    stride=layer_spec.stride,
                    padding=layer_spec.padding,
                )
                layers.append(conv)
                if layer_spec.post_activation:
                    layers.append(nn.ReLU())
            elif layer_spec.kind == "residual_pre":
                layers.append(PreActivationResidualConvBlock(in_channels, layer_spec))
            elif layer_spec.kind == "residual_post":
                layers.append(PostActivationResidualConvBlock(in_channels, layer_spec))
            elif layer_spec.kind == "maxpool":
                layers.append(torch_pooling_layer(nn.MaxPool2d, layer_spec))
            elif layer_spec.kind == "avgpool":
                layers.append(torch_pooling_layer(nn.AvgPool2d, layer_spec))
            elif layer_spec.kind == "activation" and layer_spec.activation is not None:
                layers.append(resolve_policy_activation_fn(layer_spec.activation)())
            else:
                raise ValueError(f"Unsupported CNN layer kind: {layer_spec.kind!r}")
            capture_indices.append(len(layers) - 1)
            in_channels = cnn_layer_output_channels(in_channels, layer_spec)
        return layers, tuple(capture_indices)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert either NHWC or NCHW observations into shared PPO features."""

        channels_first = self._channels_first_observations(observations)
        return self._layer_norm_activation(
            self._layer_norm(self._linear(self._cnn(channels_first)))
        )

    def convolution_activations(
        self,
        observations: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return top-level image-layer outputs for CNN visualization."""

        activations: list[tuple[str, torch.Tensor]] = []
        output = self._channels_first_observations(observations)
        capture_positions = {
            module_index: layer_index
            for layer_index, module_index in enumerate(self._cnn_capture_indices)
        }
        for module_index, layer in enumerate(self._cnn):
            if isinstance(layer, nn.Flatten):
                break
            output = layer(output)
            layer_index = capture_positions.get(module_index)
            if layer_index is None:
                continue
            layer_spec = self._conv_spec[layer_index]
            activations.append((cnn_layer_name(layer_index + 1, layer_spec.kind), output.detach()))
        return tuple(activations)

    def _channels_first_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(f"Expected a 4D observation tensor, got {tuple(observations.shape)!r}")

        if observations.shape[1:] == (self._channels, self._height, self._width):
            channels_first = observations
        elif observations.shape[1:] == (self._height, self._width, self._channels):
            channels_first = observations.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"Unexpected observation tensor shape for {type(self).__name__}: "
                f"{tuple(observations.shape)!r}"
            )

        return channels_first.float()


class FZeroXImageStateExtractor(BaseFeaturesExtractor):
    """CNN image features plus a scalar-state branch for Dict observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int | Literal["auto"] = 512,
        state_features_dim: int = 64,
        state_net_arch: tuple[int, ...] | None = None,
        state_activation: ActivationName = "relu",
        fusion_features_dim: int | None = None,
        image_projection_activation: ActivationName = "relu",
        fusion_activation: ActivationName = "relu",
        conv_profile: ConvProfile = "nature",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
        layer_norm: bool = False,
        layer_norm_activation: ActivationName | None = None,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                f"{type(self).__name__} expects a Dict observation space, "
                f"got {type(observation_space).__name__}"
            )

        image_space = observation_space.spaces.get("image")
        state_space = observation_space.spaces.get("state")
        if not isinstance(image_space, spaces.Box):
            raise ValueError(f"{type(self).__name__} requires Box key 'image'")
        if not isinstance(state_space, spaces.Box) or len(state_space.shape) != 1:
            raise ValueError(f"{type(self).__name__} requires 1D Box key 'state'")

        state_dim = int(state_space.shape[0])
        image_features_dim = (
            image_flatten_dim(
                image_space,
                extractor_name=type(self).__name__,
                conv_profile=conv_profile,
                custom_conv_layers=custom_conv_layers,
            )
            if features_dim == "auto"
            else int(features_dim)
        )
        resolved_state_net_arch = (
            (int(state_features_dim),)
            if state_net_arch is None
            else tuple(int(width) for width in state_net_arch)
        )
        state_branch_output_dim = (
            state_dim if not resolved_state_net_arch else resolved_state_net_arch[-1]
        )
        combined_features_dim = image_features_dim + state_branch_output_dim
        resolved_fusion_features_dim = (
            None if fusion_features_dim is None else int(fusion_features_dim)
        )
        output_features_dim = (
            combined_features_dim
            if resolved_fusion_features_dim is None
            else resolved_fusion_features_dim
        )
        super().__init__(observation_space, output_features_dim)

        self._state_dim = state_dim
        self._image_extractor = FZeroXObservationCnnExtractor(
            image_space,
            features_dim=features_dim,
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
            image_projection_activation=image_projection_activation,
            layer_norm=False,
        )
        self._state_mlp = _state_branch_mlp(
            input_dim=self._state_dim,
            widths=resolved_state_net_arch,
            activation=state_activation,
        )
        if resolved_fusion_features_dim is None:
            self._fusion_mlp: nn.Module = nn.Identity()
        else:
            # The fusion layer lets image and scalar-state features interact
            # before the recurrent core when a fusion width is configured.
            self._fusion_mlp = nn.Sequential(
                nn.Linear(combined_features_dim, resolved_fusion_features_dim),
                resolve_policy_activation_fn(fusion_activation)(),
            )
        self._layer_norm: nn.Module = (
            nn.LayerNorm(output_features_dim) if layer_norm else nn.Identity()
        )
        self._layer_norm_activation = _layer_norm_activation_layer(
            layer_norm=layer_norm,
            activation=layer_norm_activation,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations.get("image")
        state = observations.get("state")
        if image is None or state is None:
            raise ValueError(f"{type(self).__name__} expects observation keys 'image' and 'state'")

        state_flat = state.float().flatten(start_dim=1)
        if state_flat.shape[1] != self._state_dim:
            raise ValueError(
                f"Unexpected state vector width for {type(self).__name__}: "
                f"got {state_flat.shape[1]}, expected {self._state_dim}"
            )

        combined_features = _concat_features(
            self._image_extractor(image), self._state_mlp(state_flat)
        )
        return self._layer_norm_activation(self._layer_norm(self._fusion_mlp(combined_features)))


def _layer_norm_activation_layer(
    *,
    layer_norm: bool,
    activation: ActivationName | None,
) -> nn.Module:
    if not layer_norm or activation is None:
        return nn.Identity()
    return resolve_policy_activation_fn(activation)()


def _concat_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat((left, right), dim=1)


def _state_branch_mlp(
    *,
    input_dim: int,
    widths: tuple[int, ...],
    activation: ActivationName,
) -> nn.Module:
    if not widths:
        return nn.Identity()

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for width in widths:
        layers.append(nn.Linear(previous_dim, int(width)))
        layers.append(resolve_policy_activation_fn(activation)())
        previous_dim = int(width)
    return nn.Sequential(*layers)
