# src/rl_fzerox/core/runtime_spec/schema/policy.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from rl_fzerox.core.domain.cnn import (
    CnnActivationName,
    CnnLayerKind,
    is_activation_cnn_layer,
    normalize_cnn_layer_kind,
    validate_cnn_layer_geometry,
)
from rl_fzerox.core.policy.activations import ActivationName
from rl_fzerox.core.policy.auxiliary_state.names import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_supports_grounded_only,
)


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

        kind: CnnLayerKind = "conv"
        out_channels: PositiveInt
        kernel_size: PositiveInt
        stride: PositiveInt
        padding: int = Field(default=0, ge=0)
        post_activation: bool = True
        activation: CnnActivationName | None = None

        @field_validator("kind", mode="before")
        @classmethod
        def _normalize_kind(cls, value: object) -> CnnLayerKind:
            return normalize_cnn_layer_kind(value)

        @model_validator(mode="after")
        def _validate_layer_geometry(self) -> ExtractorConfig.CustomConvLayer:
            validate_cnn_layer_geometry(
                kind=self.kind,
                kernel_size=int(self.kernel_size),
                stride=int(self.stride),
                padding=int(self.padding),
            )
            return self

        @model_validator(mode="after")
        def _validate_activation_name(self) -> ExtractorConfig.CustomConvLayer:
            if is_activation_cnn_layer(self.kind) and self.activation is None:
                self.activation = "relu"
            return self

    conv_profile: Literal[
        "nature",
        "impala_small",
        "impala_large",
        "custom",
    ] = "nature"
    custom_conv_layers: tuple[CustomConvLayer, ...] = ()
    features_dim: PositiveInt | Literal["auto"] = 512
    image_projection_activation: ActivationName = "relu"
    state_features_dim: PositiveInt = 64
    state_net_arch: tuple[PositiveInt, ...] | None = None
    state_activation: ActivationName = "relu"
    fusion_features_dim: PositiveInt | None = None
    fusion_activation: ActivationName = "relu"
    layer_norm: bool = False
    layer_norm_activation: ActivationName | None = None

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
    air_brake_on_logit: float = 0.0
    spin_idle_logit: float = 0.0


class PolicyAuxiliaryStateLossConfig(BaseModel):
    """One supervised RAM-derived target predicted from shared policy latent."""

    model_config = ConfigDict(extra="forbid")

    name: AuxiliaryStateTargetName
    weight: float = Field(default=1.0, gt=0.0)
    grounded_only: bool = False

    @model_validator(mode="after")
    def _validate_grounded_only(self) -> PolicyAuxiliaryStateLossConfig:
        if self.grounded_only and not auxiliary_state_target_supports_grounded_only(self.name):
            raise ValueError("grounded_only is not supported for this auxiliary-state target")
        return self


class PolicyAuxiliaryStateConfig(BaseModel):
    """Optional fixed aux-head bank plus active supervised loss terms."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    head_arch: tuple[PositiveInt, ...] = (128,)
    losses: tuple[PolicyAuxiliaryStateLossConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_losses(self) -> PolicyAuxiliaryStateConfig:
        loss_names = [loss.name for loss in self.losses]
        if len(set(loss_names)) != len(loss_names):
            raise ValueError("policy.auxiliary_state.losses must not contain duplicates")
        if self.losses and not self.enabled:
            raise ValueError(
                "policy.auxiliary_state.enabled must be true when losses are configured"
            )
        return self


class PolicyConfig(BaseModel):
    """SB3 policy and feature-extractor sizes."""

    model_config = ConfigDict(extra="forbid")

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    recurrent: PolicyRecurrentConfig = Field(default_factory=PolicyRecurrentConfig)
    action_bias: PolicyActionBiasConfig = Field(default_factory=PolicyActionBiasConfig)
    auxiliary_state: PolicyAuxiliaryStateConfig = Field(default_factory=PolicyAuxiliaryStateConfig)
    activation: Literal["tanh", "relu", "gelu"] = "tanh"
    net_arch: NetArchConfig = Field(default_factory=NetArchConfig)
