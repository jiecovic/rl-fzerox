# src/rl_fzerox/core/runtime_spec/schema/policy.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class NetArchConfig(BaseModel):
    """SB3 actor/critic head sizes after the shared CNN extractor."""

    model_config = ConfigDict(extra="forbid")

    pi: tuple[PositiveInt, ...] = (256, 256)
    vf: tuple[PositiveInt, ...] = (256, 256)


class ExtractorConfig(BaseModel):
    """Shared feature-extractor settings for SB3 policies."""

    model_config = ConfigDict(extra="forbid")

    class CustomConvLayer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        out_channels: PositiveInt
        kernel_size: PositiveInt
        stride: PositiveInt
        padding: int = Field(default=0, ge=0)

    conv_profile: Literal[
        "auto",
        "nature",
        "nature_32_64_128",
        "nature_wide",
        "nature_extra_k3",
        "compact_deep",
        "compact_bottleneck",
        "tiny_256",
        "custom",
    ] = "auto"
    custom_conv_layers: tuple[CustomConvLayer, ...] = ()
    features_dim: PositiveInt | Literal["auto"] = 512
    state_features_dim: PositiveInt = 64
    state_net_arch: tuple[PositiveInt, ...] | None = None
    fusion_features_dim: PositiveInt | None = None
    layer_norm: bool = False

    @model_validator(mode="after")
    def _validate_custom_conv_layers(self) -> ExtractorConfig:
        if self.conv_profile == "custom" and not self.custom_conv_layers:
            raise ValueError("policy.extractor.custom_conv_layers must not be empty")
        return self

    def resolved_state_net_arch(self) -> tuple[int, ...]:
        """Return the canonical state MLP widths for runtime/extractor code."""

        if self.state_net_arch is None:
            return (int(self.state_features_dim),)
        return tuple(int(width) for width in self.state_net_arch)


class PolicyRecurrentConfig(BaseModel):
    """Optional LSTM settings used only by recurrent PPO-family policies."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    hidden_size: PositiveInt = 512
    n_lstm_layers: PositiveInt = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True


class PolicyActionBiasConfig(BaseModel):
    """Optional initial policy-head logit nudges for discrete action branches."""

    model_config = ConfigDict(extra="forbid")

    gas_on_logit: float = 0.0


class PolicyConfig(BaseModel):
    """SB3 policy and feature-extractor sizes."""

    model_config = ConfigDict(extra="forbid")

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    recurrent: PolicyRecurrentConfig = Field(default_factory=PolicyRecurrentConfig)
    action_bias: PolicyActionBiasConfig = Field(default_factory=PolicyActionBiasConfig)
    activation: Literal["tanh", "relu", "gelu"] = "tanh"
    net_arch: NetArchConfig = Field(default_factory=NetArchConfig)
