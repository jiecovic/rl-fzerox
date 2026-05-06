"""Canonical manager-owned run-spec surface.

The package root is intentionally small:
- ``run`` owns the top-level immutable SQLite snapshot model
- ``sections`` owns the per-tab/per-surface section models
- ``common`` owns shared enums and small value types used by both
"""

from rl_fzerox.core.manager.run_spec.common import (
    ConfigVersion,
    ConvProfile,
    ObservationPreset,
)
from rl_fzerox.core.manager.run_spec.run import (
    ManagedRunConfig,
    default_managed_run_config,
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
