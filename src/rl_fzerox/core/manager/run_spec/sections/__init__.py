# src/rl_fzerox/core/manager/run_spec/sections/__init__.py
"""Section models for the manager-owned run-spec surface.

Each module in this package maps cleanly to one editable surface in the
frontend/run-manager UI. Keeping them here avoids another flat bucket at the
``core.manager`` package root while preserving a narrow public API.
"""

from rl_fzerox.core.manager.run_spec.sections.action import ManagedActionConfig
from rl_fzerox.core.manager.run_spec.sections.environment import ManagedEnvironmentConfig
from rl_fzerox.core.manager.run_spec.sections.observation import (
    ManagedObservationConfig,
    ManagedStateComponentConfig,
    ManagedStateFeatureDropoutConfig,
    default_state_components,
    default_state_feature_dropouts,
    managed_state_component_feature_names,
)
from rl_fzerox.core.manager.run_spec.sections.policy import ManagedPolicyConfig
from rl_fzerox.core.manager.run_spec.sections.reward import ManagedRewardConfig
from rl_fzerox.core.manager.run_spec.sections.tracks import (
    ManagedTracksConfig,
    default_selected_course_ids,
)
from rl_fzerox.core.manager.run_spec.sections.training import ManagedTrainConfig
from rl_fzerox.core.manager.run_spec.sections.vehicle import ManagedVehicleConfig

__all__ = [
    "ManagedActionConfig",
    "ManagedEnvironmentConfig",
    "ManagedObservationConfig",
    "ManagedPolicyConfig",
    "ManagedRewardConfig",
    "ManagedStateComponentConfig",
    "ManagedStateFeatureDropoutConfig",
    "ManagedTracksConfig",
    "ManagedTrainConfig",
    "ManagedVehicleConfig",
    "default_selected_course_ids",
    "default_state_components",
    "default_state_feature_dropouts",
    "managed_state_component_feature_names",
]
