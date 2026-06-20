# src/rl_fzerox/core/training/session/callbacks/track_sampling/__init__.py
"""Dynamic course-sampling controller facade."""

from __future__ import annotations

from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    TrackSamplingAltBaseline,
    alt_baseline_signature,
    apply_alt_baselines_to_track_sampling,
    stable_entry_alt_baseline_key,
    strip_alt_baselines,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.artifacts import (
    TrackSamplingMaterializedArtifact,
    materialized_track_sampling_artifacts,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.controller import (
    StepBalancedTrackSamplingController,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.courses import (
    GeneratedCourseMetadata,
    TrackSamplingCourseEntry,
    resolve_track_sampling_courses_from_entries,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.deficit import (
    DEFICIT_QUEUE_SETTINGS,
    DeficitBudgetTrackSamplingController,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.persistence import (
    TrackSamplingRuntimePersistence,
    deficit_budget_scheduler_state_json,
    file_track_sampling_runtime_persistence,
    load_deficit_budget_scheduler_state_json,
    load_track_sampling_runtime_state,
    load_track_sampling_runtime_state_json,
    save_track_sampling_runtime_state,
    track_sampling_runtime_state_json,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    DeficitBudgetCourseSchedulerState,
    DeficitBudgetSchedulerState,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    replace_runtime_generation,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.weights import (
    adaptive_confidence_bonus,
    adaptive_difficulty_bonus,
    adaptive_target_bonus,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.x_cup_rotation import (
    XCupRotationManager,
    XCupRotationUpdate,
)

__all__ = (
    "StepBalancedTrackSamplingController",
    "DEFICIT_QUEUE_SETTINGS",
    "DeficitBudgetTrackSamplingController",
    "DeficitBudgetCourseSchedulerState",
    "DeficitBudgetSchedulerState",
    "TrackSamplingRuntimeEntry",
    "GeneratedCourseMetadata",
    "TrackSamplingMaterializedArtifact",
    "TrackSamplingAltBaseline",
    "TrackSamplingCourseEntry",
    "TrackSamplingRuntimePersistence",
    "TrackSamplingRuntimeState",
    "XCupRotationManager",
    "XCupRotationUpdate",
    "adaptive_confidence_bonus",
    "adaptive_difficulty_bonus",
    "adaptive_target_bonus",
    "alt_baseline_signature",
    "apply_alt_baselines_to_track_sampling",
    "deficit_budget_scheduler_state_json",
    "file_track_sampling_runtime_persistence",
    "load_deficit_budget_scheduler_state_json",
    "load_track_sampling_runtime_state",
    "load_track_sampling_runtime_state_json",
    "materialized_track_sampling_artifacts",
    "replace_runtime_generation",
    "resolve_track_sampling_courses_from_entries",
    "save_track_sampling_runtime_state",
    "stable_entry_alt_baseline_key",
    "strip_alt_baselines",
    "track_sampling_runtime_state_json",
)
