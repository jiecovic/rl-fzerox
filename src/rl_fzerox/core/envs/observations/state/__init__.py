# src/rl_fzerox/core/envs/observations/state/__init__.py
"""State-vector observation facade.

The public surface here exposes state feature specs, telemetry conversion, and
action-history feature naming. Component-level feature ownership is kept in the
`components` package.
"""
from __future__ import annotations

from rl_fzerox.core.domain.observations import StateComponentsSettings

from .api import (
    action_history_feature_names,
    action_history_settings_for_observation,
    state_feature_count,
    state_feature_names,
    state_vector_spec,
    telemetry_state_vector,
)
from .types import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
    ObservationMode,
    StateFeature,
    StateVectorSpec,
)

__all__ = [
    "ActionHistoryControl",
    "OBSERVATION_STATE_DEFAULTS",
    "ObservationMode",
    "StateComponentsSettings",
    "StateFeature",
    "StateVectorSpec",
    "action_history_feature_names",
    "action_history_settings_for_observation",
    "state_feature_count",
    "state_feature_names",
    "state_vector_spec",
    "telemetry_state_vector",
]
