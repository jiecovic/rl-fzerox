# src/rl_fzerox/core/envs/state_observation/__init__.py
from __future__ import annotations

from rl_fzerox.core.domain.observation_components import StateComponentsSettings
from rl_fzerox.core.envs.state_observation.api import (
    action_history_feature_names,
    action_history_settings_for_observation,
    state_feature_count,
    state_feature_names,
    state_vector_spec,
    telemetry_state_vector,
)
from rl_fzerox.core.envs.state_observation.legacy import (
    DEFAULT_STATE_VECTOR_SPEC,
    LEGACY_STATE_VECTOR_EXPORTS,
)
from rl_fzerox.core.envs.state_observation.types import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
    ObservationCourseContext,
    ObservationGroundEffectContext,
    ObservationMode,
    ObservationStateProfile,
    StateFeature,
    StateVectorSpec,
)

# Legacy exports kept for older call sites and run-manifest loading. New code
# should depend on OBSERVATION_STATE_DEFAULTS or explicit StateVectorSpec values.
DEFAULT_OBSERVATION_STATE_PROFILE = OBSERVATION_STATE_DEFAULTS.state_profile
DEFAULT_OBSERVATION_COURSE_CONTEXT = OBSERVATION_STATE_DEFAULTS.course_context
DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT = OBSERVATION_STATE_DEFAULTS.ground_effect_context
DEFAULT_ACTION_HISTORY_LEN = OBSERVATION_STATE_DEFAULTS.action_history_len
DEFAULT_ACTION_HISTORY_CONTROLS = OBSERVATION_STATE_DEFAULTS.action_history_controls
STATE_VECTOR_SPEC = LEGACY_STATE_VECTOR_EXPORTS.spec
STATE_FEATURE_NAMES = LEGACY_STATE_VECTOR_EXPORTS.names
STATE_FEATURE_COUNT = LEGACY_STATE_VECTOR_EXPORTS.count
STATE_SPEED_NORMALIZER_KPH = LEGACY_STATE_VECTOR_EXPORTS.speed_normalizer_kph
LEAN_DOUBLE_TAP_WINDOW_FRAMES = LEGACY_STATE_VECTOR_EXPORTS.lean_tap_guard_frames
RECENT_BOOST_PRESSURE_WINDOW_FRAMES = LEGACY_STATE_VECTOR_EXPORTS.recent_boost_window_frames
RECENT_STEER_PRESSURE_WINDOW_FRAMES = LEGACY_STATE_VECTOR_EXPORTS.recent_steer_window_frames
STATE_FEATURE_LOW = LEGACY_STATE_VECTOR_EXPORTS.low
STATE_FEATURE_HIGH = LEGACY_STATE_VECTOR_EXPORTS.high

__all__ = [
    "ActionHistoryControl",
    "DEFAULT_ACTION_HISTORY_CONTROLS",
    "DEFAULT_ACTION_HISTORY_LEN",
    "DEFAULT_OBSERVATION_COURSE_CONTEXT",
    "DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT",
    "DEFAULT_OBSERVATION_STATE_PROFILE",
    "DEFAULT_STATE_VECTOR_SPEC",
    "LEAN_DOUBLE_TAP_WINDOW_FRAMES",
    "ObservationCourseContext",
    "ObservationGroundEffectContext",
    "ObservationMode",
    "ObservationStateProfile",
    "RECENT_BOOST_PRESSURE_WINDOW_FRAMES",
    "RECENT_STEER_PRESSURE_WINDOW_FRAMES",
    "STATE_FEATURE_COUNT",
    "STATE_FEATURE_HIGH",
    "STATE_FEATURE_LOW",
    "STATE_FEATURE_NAMES",
    "STATE_SPEED_NORMALIZER_KPH",
    "STATE_VECTOR_SPEC",
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
