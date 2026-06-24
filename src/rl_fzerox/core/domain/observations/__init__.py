# src/rl_fzerox/core/domain/observations/__init__.py
"""Observation-domain component and image geometry vocabulary."""

from __future__ import annotations

from rl_fzerox.core.domain.observations.components import (
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    StateComponentFeaturePolicy,
    StateComponentsSettings,
    TrackPositionProgressSourceName,
    default_excluded_state_feature_names,
    state_feature_default_enabled,
)
from rl_fzerox.core.domain.observations.image import (
    OBSERVATION_IMAGE_GEOMETRY,
    OBSERVATION_PRESET_GEOMETRY_BY_NAME,
    CustomResolutionBounds,
    CustomResolutionChoice,
    ObservationImageGeometry,
    ObservationPresetGeometry,
    ObservationPresetName,
    ObservationRendererName,
    ObservationResizeFilter,
    ObservationResolutionConfig,
    ObservationResolutionMode,
    ObservationSourceGeometry,
    PresetResolutionChoice,
    SourceCropResolutionChoice,
    preset_geometry,
    resolve_observation_geometry,
    source_crop_geometry,
)

__all__ = (
    "ActionHistoryControlName",
    "CustomResolutionBounds",
    "CustomResolutionChoice",
    "OBSERVATION_IMAGE_GEOMETRY",
    "OBSERVATION_PRESET_GEOMETRY_BY_NAME",
    "ObservationCourseContextName",
    "ObservationImageGeometry",
    "ObservationPresetGeometry",
    "ObservationPresetName",
    "ObservationRendererName",
    "ObservationResolutionConfig",
    "ObservationResolutionMode",
    "ObservationResizeFilter",
    "ObservationSourceGeometry",
    "ObservationStateComponentName",
    "ObservationStateComponentSettings",
    "PresetResolutionChoice",
    "SourceCropResolutionChoice",
    "StateComponentFeaturePolicy",
    "StateComponentsSettings",
    "TrackPositionProgressSourceName",
    "default_excluded_state_feature_names",
    "preset_geometry",
    "resolve_observation_geometry",
    "source_crop_geometry",
    "state_feature_default_enabled",
)
