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

from rl_fzerox.core.domain.policy import (
    CnnActivationName,
    CnnLayerKind,
)
from rl_fzerox.core.manager.run_spec.common import (
    ActivationName,
    ConvProfile,
    FeatureDim,
)
from rl_fzerox.core.policy.auxiliary_state.names import AuxiliaryStateTargetName
from rl_fzerox.core.policy.schema_validation import (
    default_policy_cnn_layer_activation,
    normalize_policy_cnn_layer_kind,
    validate_policy_auxiliary_grounded_only,
    validate_policy_auxiliary_losses,
    validate_policy_cnn_layer_geometry,
    validate_policy_custom_conv_layers_present,
)


class ManagedPolicyAuxiliaryStateLossConfig(BaseModel):
    """One optional aux prediction loss term exposed for advanced experiments."""

    model_config = ConfigDict(extra="forbid")

    name: AuxiliaryStateTargetName
    weight: float = Field(default=1.0, gt=0.0)
    grounded_only: bool = False

    @model_validator(mode="after")
    def _validate_grounded_only(self) -> ManagedPolicyAuxiliaryStateLossConfig:
        validate_policy_auxiliary_grounded_only(
            name=self.name,
            grounded_only=self.grounded_only,
        )
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
            return normalize_policy_cnn_layer_kind(value)

        @model_validator(mode="after")
        def _validate_layer_geometry(self) -> ManagedPolicyConfig.CustomConvLayer:
            validate_policy_cnn_layer_geometry(
                kind=self.kind,
                kernel_size=int(self.kernel_size),
                stride=int(self.stride),
                padding=int(self.padding),
            )
            return self

        @model_validator(mode="after")
        def _validate_activation_name(self) -> ManagedPolicyConfig.CustomConvLayer:
            self.activation = default_policy_cnn_layer_activation(
                kind=self.kind,
                activation=self.activation,
            )
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
    air_brake_on_logit: float = 0.0
    spin_idle_logit: float = 0.0
    auxiliary_state_enabled: bool = False
    auxiliary_state_head_arch: tuple[PositiveInt, ...] = (128,)
    auxiliary_state_losses: tuple[ManagedPolicyAuxiliaryStateLossConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_custom_conv_layers(self) -> ManagedPolicyConfig:
        validate_policy_custom_conv_layers_present(
            conv_profile=self.conv_profile,
            has_custom_layers=bool(self.custom_conv_layers),
            message="policy.custom_conv_layers must not be empty for conv_profile=custom",
        )
        validate_policy_auxiliary_losses(
            loss_names=tuple(loss.name for loss in self.auxiliary_state_losses),
            enabled=self.auxiliary_state_enabled,
            duplicate_message="policy.auxiliary_state_losses must not contain duplicates",
            disabled_message=(
                "policy.auxiliary_state_enabled must be true when "
                "auxiliary_state_losses are configured"
            ),
        )
        return self


def default_custom_conv_layers() -> tuple[ManagedPolicyConfig.CustomConvLayer, ...]:
    """Return a sensible custom-CNN starting point for the manager."""

    return (
        ManagedPolicyConfig.CustomConvLayer(out_channels=32, kernel_size=8, stride=4, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=64, kernel_size=4, stride=2, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=128, kernel_size=3, stride=1, padding=0),
    )
