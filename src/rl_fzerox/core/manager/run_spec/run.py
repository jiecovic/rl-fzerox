# src/rl_fzerox/core/manager/run_spec/run.py
"""Root manager-owned run-spec model.

This module assembles the section models from ``run_spec.sections`` into the
single SQLite-backed config snapshot that the run manager stores and version
controls through the database.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from rl_fzerox.core.manager.run_spec.common import (
    ConfigVersion,
    ConvProfile,
    ObservationPreset,
)
from rl_fzerox.core.manager.run_spec.sections import (
    ManagedActionConfig,
    ManagedEnvironmentConfig,
    ManagedObservationConfig,
    ManagedPolicyConfig,
    ManagedRewardConfig,
    ManagedStateComponentConfig,
    ManagedStateFeatureDropoutConfig,
    ManagedTracksConfig,
    ManagedTrainConfig,
    ManagedVehicleConfig,
    default_selected_course_ids,
    default_state_components,
    default_state_feature_dropouts,
    managed_state_component_feature_names,
)


class ManagedRunConfig(BaseModel):
    """DB-owned immutable config snapshot for one managed run."""

    model_config = ConfigDict(extra="forbid")

    version: ConfigVersion = 1
    seed: int = 123
    preset_name: str = "all-cups recurrent PPO"
    tracks: ManagedTracksConfig = Field(default_factory=ManagedTracksConfig)
    vehicle: ManagedVehicleConfig = Field(default_factory=ManagedVehicleConfig)
    action: ManagedActionConfig = Field(default_factory=ManagedActionConfig)
    environment: ManagedEnvironmentConfig = Field(default_factory=ManagedEnvironmentConfig)
    train: ManagedTrainConfig = Field(default_factory=ManagedTrainConfig)
    observation: ManagedObservationConfig = Field(default_factory=ManagedObservationConfig)
    policy: ManagedPolicyConfig = Field(default_factory=ManagedPolicyConfig)
    reward: ManagedRewardConfig = Field(default_factory=ManagedRewardConfig)

    @model_validator(mode="after")
    def _validate_custom_conv_geometry(self) -> ManagedRunConfig:
        from rl_fzerox.core.policy.extractors import (
            ensure_conv_spec_fits_geometry,
            resolve_conv_spec,
        )

        height, width = self.observation.image_geometry()
        if self.observation.resolution_mode == "custom" and self.policy.conv_profile == "auto":
            raise ValueError(
                "policy.conv_profile='auto' only supports named observation presets; "
                "choose an explicit conv_profile for custom image resolutions"
            )
        conv_spec = resolve_conv_spec(
            (height, width),
            conv_profile=self.policy.conv_profile,
            custom_conv_layers=tuple(
                layer.model_dump(mode="python") for layer in self.policy.custom_conv_layers
            ),
        )
        ensure_conv_spec_fits_geometry(
            height=height,
            width=width,
            conv_spec=conv_spec,
            profile_name=self.policy.conv_profile,
        )
        active_features = managed_state_component_feature_names(
            self.observation.state_components,
            independent_lean_buttons=self.action.lean_output_mode == "independent_buttons",
        )
        unknown_features = [
            feature.name
            for feature in self.observation.state_feature_dropouts
            if feature.name not in active_features
        ]
        if unknown_features:
            joined = ", ".join(sorted(unknown_features))
            raise ValueError(
                "observation.state_feature_dropouts must reference active state features: "
                f"{joined}"
            )
        return self


def default_managed_run_config() -> ManagedRunConfig:
    """Return the first manager preset without reading any YAML files."""

    return ManagedRunConfig()


__all__ = [
    "ConfigVersion",
    "ConvProfile",
    "ManagedActionConfig",
    "ManagedEnvironmentConfig",
    "ManagedObservationConfig",
    "ManagedPolicyConfig",
    "ManagedRewardConfig",
    "ManagedRunConfig",
    "ManagedStateComponentConfig",
    "ManagedStateFeatureDropoutConfig",
    "ManagedTracksConfig",
    "ManagedTrainConfig",
    "ManagedVehicleConfig",
    "ObservationPreset",
    "default_managed_run_config",
    "default_selected_course_ids",
    "default_state_components",
    "default_state_feature_dropouts",
    "managed_state_component_feature_names",
]
