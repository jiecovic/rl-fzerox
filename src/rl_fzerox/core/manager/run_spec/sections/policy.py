# src/rl_fzerox/core/manager/run_spec/sections/policy.py
"""Policy/extractor section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.manager.run_spec.common import (
    ActivationName,
    ConvProfile,
    FeatureDim,
)


class ManagedPolicyConfig(BaseModel):
    """Policy architecture knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    class CustomConvLayer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        out_channels: PositiveInt
        kernel_size: PositiveInt
        stride: PositiveInt
        padding: NonNegativeInt = 0

    conv_profile: ConvProfile = "nature"
    custom_conv_layers: tuple[CustomConvLayer, ...] = Field(
        default_factory=lambda: default_custom_conv_layers()
    )
    features_dim: FeatureDim = "auto"
    state_net_arch: tuple[PositiveInt, ...] = (64,)
    fusion_features_dim: PositiveInt = 768
    layer_norm: bool = True
    activation: ActivationName = "relu"
    recurrent_enabled: bool = True
    recurrent_hidden_size: PositiveInt = 256
    recurrent_n_lstm_layers: PositiveInt = 1
    recurrent_shared_lstm: bool = False
    recurrent_enable_critic_lstm: bool = True
    pi_net_arch: tuple[PositiveInt, ...] = (256, 128)
    vf_net_arch: tuple[PositiveInt, ...] = (256, 128)
    gas_on_logit: float = 0.0

    @model_validator(mode="after")
    def _validate_custom_conv_layers(self) -> ManagedPolicyConfig:
        if self.conv_profile == "custom" and not self.custom_conv_layers:
            raise ValueError("policy.custom_conv_layers must not be empty for conv_profile=custom")
        return self


def default_custom_conv_layers() -> tuple[ManagedPolicyConfig.CustomConvLayer, ...]:
    """Return a sensible custom-CNN starting point for the manager."""

    return (
        ManagedPolicyConfig.CustomConvLayer(out_channels=32, kernel_size=8, stride=4, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=64, kernel_size=4, stride=2, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=128, kernel_size=3, stride=1, padding=0),
    )
