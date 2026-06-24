# src/rl_fzerox/core/manager/architecture/preview/params.py
from __future__ import annotations

from rl_fzerox.core.domain.policy import (
    is_activation_cnn_layer,
    is_pooling_cnn_layer,
    is_residual_cnn_layer,
)
from rl_fzerox.core.manager.architecture.models import ConvLayerPreview
from rl_fzerox.core.manager.run_spec import ConvProfile, ManagedRunConfig
from rl_fzerox.core.policy.extractors import (
    ConvLayerSpec,
    cnn_layer_name,
    cnn_layer_output_channels,
    cnn_layer_output_shape,
    resolve_conv_spec,
)


def conv_layer_previews(
    *,
    height: int,
    width: int,
    channels: int,
    conv_profile: ConvProfile,
    custom_conv_layers: tuple[dict[str, object], ...] | None = None,
) -> tuple[tuple[ConvLayerPreview, ...], int]:
    layers: list[ConvLayerPreview] = []
    in_channels = channels
    output_height = height
    output_width = width
    for index, layer in enumerate(
        resolve_conv_spec(
            (height, width),
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
        ),
        start=1,
    ):
        kernel_size = int(layer.kernel_size[0])
        stride = int(layer.stride[0])
        padding = int(layer.padding[0])
        input_height = output_height
        input_width = output_width
        output_height, output_width = cnn_layer_output_shape(
            height=input_height,
            width=input_width,
            layer_spec=layer,
        )
        dropped_height = dropped_trailing_pixels(
            input_size=input_height,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_height,
        )
        dropped_width = dropped_trailing_pixels(
            input_size=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_width,
        )
        output_channels = cnn_layer_output_channels(in_channels, layer)
        params = cnn_layer_params(in_channels=in_channels, layer=layer)
        layers.append(
            ConvLayerPreview(
                name=cnn_layer_name(index, layer.kind),
                kind=layer.kind,
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                post_activation=layer.post_activation,
                activation=layer.activation,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                dropped_height=dropped_height,
                dropped_width=dropped_width,
                params=params,
            )
        )
        in_channels = output_channels
    return tuple(layers), in_channels * output_height * output_width


def recurrent_param_count(config: ManagedRunConfig, input_dim: int) -> int:
    if not config.policy.recurrent_enabled:
        return 0
    actor_params = lstm_params(
        input_dim=input_dim,
        hidden_size=int(config.policy.recurrent_hidden_size),
        layers=int(config.policy.recurrent_n_lstm_layers),
    )
    if config.policy.recurrent_shared_lstm or not config.policy.recurrent_enable_critic_lstm:
        return actor_params
    return actor_params * 2


def linear_params(in_features: int, out_features: int) -> int:
    if out_features <= 0:
        return 0
    return (in_features * out_features) + out_features


def conv_params(*, in_channels: int, out_channels: int, kernel_size: int) -> int:
    return (in_channels * out_channels * kernel_size * kernel_size) + out_channels


def cnn_layer_params(*, in_channels: int, layer: ConvLayerSpec) -> int:
    if is_pooling_cnn_layer(layer.kind):
        return 0
    if is_activation_cnn_layer(layer.kind):
        return 0
    kernel_size = int(layer.kernel_size[0])
    params = conv_params(
        in_channels=in_channels,
        out_channels=layer.out_channels,
        kernel_size=kernel_size,
    )
    if not is_residual_cnn_layer(layer.kind):
        return params

    params += conv_params(
        in_channels=layer.out_channels,
        out_channels=layer.out_channels,
        kernel_size=kernel_size,
    )
    if in_channels != layer.out_channels or layer.stride != (1, 1):
        params += conv_params(
            in_channels=in_channels,
            out_channels=layer.out_channels,
            kernel_size=1,
        )
    return params


def dropped_trailing_pixels(
    *,
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_size: int,
) -> int:
    last_window_end = ((output_size - 1) * stride) - padding + kernel_size
    return max(0, input_size - min(input_size, last_window_end))


def mlp_params(input_dim: int, layers: tuple[int, ...]) -> int:
    total = 0
    current = input_dim
    for layer_width in layers:
        width = int(layer_width)
        total += linear_params(current, width)
        current = width
    return total


def lstm_params(*, input_dim: int, hidden_size: int, layers: int) -> int:
    total = 0
    current_input = input_dim
    for _ in range(layers):
        total += 4 * hidden_size * (current_input + hidden_size + 2)
        current_input = hidden_size
    return total
