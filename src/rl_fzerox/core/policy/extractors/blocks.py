# src/rl_fzerox/core/policy/extractors/blocks.py
"""Residual CNN blocks used by F-Zero X policy feature extractors."""

from __future__ import annotations

import torch
from torch import nn

from rl_fzerox.core.policy.extractors.specs import ConvLayerSpec


class PostActivationResidualConvBlock(nn.Module):
    """Residual block matching the custom-CNN block behavior used so far."""

    def __init__(
        self,
        in_channels: int,
        layer_spec: ConvLayerSpec,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=layer_spec.stride,
            padding=layer_spec.padding,
        )
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
            layer_spec.out_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=(1, 1),
            padding=layer_spec.padding,
        )
        self.projection = _residual_projection(in_channels, layer_spec)
        self.output_activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.activation(self.conv1(inputs))
        output = self.conv2(output)
        return self.output_activation(output + residual)


class PreActivationResidualConvBlock(nn.Module):
    """IMPALA-style residual block with an identity-preserving skip path."""

    def __init__(
        self,
        in_channels: int,
        layer_spec: ConvLayerSpec,
    ) -> None:
        super().__init__()
        self.activation1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=layer_spec.stride,
            padding=layer_spec.padding,
        )
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            layer_spec.out_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=(1, 1),
            padding=layer_spec.padding,
        )
        self.projection = _residual_projection(in_channels, layer_spec)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.conv1(self.activation1(inputs))
        output = self.conv2(self.activation2(output))
        return output + residual


def torch_pooling_layer(
    pool_type: type[nn.MaxPool2d] | type[nn.AvgPool2d],
    layer_spec: ConvLayerSpec,
) -> nn.Module:
    return pool_type(
        kernel_size=layer_spec.kernel_size,
        stride=layer_spec.stride,
        padding=layer_spec.padding,
    )


def _residual_projection(in_channels: int, layer_spec: ConvLayerSpec) -> nn.Module:
    if in_channels == layer_spec.out_channels and layer_spec.stride == (1, 1):
        return nn.Identity()
    return nn.Conv2d(
        in_channels,
        layer_spec.out_channels,
        kernel_size=(1, 1),
        stride=layer_spec.stride,
        padding=(0, 0),
    )
