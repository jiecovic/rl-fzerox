# src/rl_fzerox/core/config/schema_models/policy.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class NetArchConfig(BaseModel):
    """SB3 actor/critic head sizes after the shared CNN extractor."""

    model_config = ConfigDict(extra="forbid")

    pi: tuple[PositiveInt, ...] = (256, 256)
    vf: tuple[PositiveInt, ...] = (256, 256)


class ExtractorConfig(BaseModel):
    """Shared feature-extractor settings for SB3 policies."""

    model_config = ConfigDict(extra="forbid")

    conv_profile: Literal[
        "auto",
        "nature",
        "nature_32_64_128",
        "nature_wide",
        "nature_extra_k3",
        "compact_deep",
        "compact_bottleneck",
        "tiny_256",
    ] = "auto"
    features_dim: PositiveInt | Literal["auto"] = 512
    state_features_dim: PositiveInt = 64
    fusion_features_dim: PositiveInt | None = None
    layer_norm: bool = False


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
    activation: Literal["tanh", "relu"] = "tanh"
    net_arch: NetArchConfig = Field(default_factory=NetArchConfig)
