# src/rl_fzerox/core/manager/projection/__init__.py
"""Projection from manager-owned run specs into launch/runtime training config."""

from rl_fzerox.core.manager.projection.actions import (
    build_action_data,
    continuous_action_axes,
    discrete_action_axes,
)
from rl_fzerox.core.manager.projection.compat import (
    assert_managed_fork_compatible,
    fork_compatibility_signature,
)
from rl_fzerox.core.manager.projection.launches import (
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)
from rl_fzerox.core.manager.projection.observations import (
    build_observation_data,
    build_state_feature_dropout_groups,
    component_feature_names,
    fork_observation_signature,
)
from rl_fzerox.core.manager.projection.policy import (
    build_policy_data,
    fork_policy_signature,
)
from rl_fzerox.core.manager.projection.tracks import build_track_sampling_data
from rl_fzerox.core.manager.projection.watch import (
    lineage_frame_offset_for_run,
    managed_watch_train_config,
)

__all__ = [
    "build_action_data",
    "build_managed_fork_train_app_config",
    "build_managed_resume_train_app_config",
    "build_managed_train_app_config",
    "build_observation_data",
    "build_policy_data",
    "build_state_feature_dropout_groups",
    "build_track_sampling_data",
    "component_feature_names",
    "continuous_action_axes",
    "discrete_action_axes",
    "assert_managed_fork_compatible",
    "fork_compatibility_signature",
    "fork_observation_signature",
    "fork_policy_signature",
    "lineage_frame_offset_for_run",
    "managed_watch_train_config",
]
