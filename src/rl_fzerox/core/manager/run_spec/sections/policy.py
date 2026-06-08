# src/rl_fzerox/core/manager/run_spec/sections/policy.py
"""Policy/extractor section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from rl_fzerox.core.domain.cnn import (
    CnnActivationName,
    CnnLayerKind,
    is_activation_cnn_layer,
    normalize_cnn_layer_kind,
    validate_cnn_layer_geometry,
)
from rl_fzerox.core.manager.run_spec.common import (
    ActivationName,
    ConvProfile,
    FeatureDim,
)
from rl_fzerox.core.policy.auxiliary_state.names import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_supports_grounded_only,
)


class ManagedPolicyAuxiliaryStateLossConfig(BaseModel):
    """One optional aux prediction loss term exposed for advanced experiments."""

    model_config = ConfigDict(extra="forbid")

    name: AuxiliaryStateTargetName
    weight: float = Field(default=1.0, gt=0.0)
    grounded_only: bool = False

    @model_validator(mode="after")
    def _validate_grounded_only(self) -> ManagedPolicyAuxiliaryStateLossConfig:
        if self.grounded_only and not auxiliary_state_target_supports_grounded_only(self.name):
            raise ValueError("grounded_only is not supported for this auxiliary-state target")
        return self


class ManagedPolicyConfig(BaseModel):
    """Policy architecture knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    class CustomConvLayer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        kind: CnnLayerKind = "conv"
        out_channels: PositiveInt
        kernel_size: PositiveInt
        stride: PositiveInt
        padding: NonNegativeInt = 0
        post_activation: bool = True
        activation: CnnActivationName | None = None

        @field_validator("kind", mode="before")
        @classmethod
        def _normalize_kind(cls, value: object) -> CnnLayerKind:
            return normalize_cnn_layer_kind(value)

        @model_validator(mode="after")
        def _validate_layer_geometry(self) -> ManagedPolicyConfig.CustomConvLayer:
            validate_cnn_layer_geometry(
                kind=self.kind,
                kernel_size=int(self.kernel_size),
                stride=int(self.stride),
                padding=int(self.padding),
            )
            return self

        @model_validator(mode="after")
        def _validate_activation_name(self) -> ManagedPolicyConfig.CustomConvLayer:
            if is_activation_cnn_layer(self.kind) and self.activation is None:
                self.activation = "relu"
            return self

    conv_profile: ConvProfile = "nature"
    custom_conv_layers: tuple[CustomConvLayer, ...] = Field(
        default_factory=lambda: default_custom_conv_layers()
    )
    features_dim: FeatureDim = "auto"
    image_projection_activation: ActivationName = "relu"
    state_net_arch: tuple[PositiveInt, ...] = (64,)
    state_activation: ActivationName = "relu"
    fusion_features_dim: PositiveInt | None = 768
    fusion_activation: ActivationName = "relu"
    layer_norm: bool = True
    layer_norm_activation: ActivationName | None = None
    activation: ActivationName = "relu"
    recurrent_enabled: bool = True
    recurrent_hidden_size: PositiveInt = 256
    recurrent_n_lstm_layers: PositiveInt = 1
    recurrent_shared_lstm: bool = False
    recurrent_enable_critic_lstm: bool = True
    pi_net_arch: tuple[PositiveInt, ...] = (256, 128)
    vf_net_arch: tuple[PositiveInt, ...] = (256, 128)
    gas_on_logit: float = 0.0
    spin_idle_logit: float = 0.0
    auxiliary_state_enabled: bool = False
    auxiliary_state_head_arch: tuple[PositiveInt, ...] = (128,)
    auxiliary_state_losses: tuple[ManagedPolicyAuxiliaryStateLossConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_custom_conv_layers(self) -> ManagedPolicyConfig:
        if self.conv_profile == "custom" and not self.custom_conv_layers:
            raise ValueError("policy.custom_conv_layers must not be empty for conv_profile=custom")
        loss_names = [loss.name for loss in self.auxiliary_state_losses]
        if len(set(loss_names)) != len(loss_names):
            raise ValueError("policy.auxiliary_state_losses must not contain duplicates")
        if self.auxiliary_state_losses and not self.auxiliary_state_enabled:
            raise ValueError(
                "policy.auxiliary_state_enabled must be true when "
                "auxiliary_state_losses are configured"
            )
        return self


def default_custom_conv_layers() -> tuple[ManagedPolicyConfig.CustomConvLayer, ...]:
    """Return a sensible custom-CNN starting point for the manager."""

    return (
        ManagedPolicyConfig.CustomConvLayer(out_channels=32, kernel_size=8, stride=4, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=64, kernel_size=4, stride=2, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=128, kernel_size=3, stride=1, padding=0),
    )
