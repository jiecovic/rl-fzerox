# src/rl_fzerox/core/envs/engine/reset/track_sampling/__init__.py
"""Track reset sampling facade for multi-target training.

This package selects the next target for an env reset and keeps small caches
for baseline states. Resolved course definitions and run-manager target
configuration arrive as runtime inputs.
"""

from __future__ import annotations

from rl_fzerox.core.envs.engine.reset.track_sampling.cache import TrackBaselineCache
from rl_fzerox.core.envs.engine.reset.track_sampling.models import (
    TRACK_BASELINE_CACHE_LIMITS,
    TRACK_SAMPLING_LIMITS,
    SelectedTrack,
    TrackBaselineCacheLimits,
    TrackSamplingDeficitLane,
    TrackSamplingLimits,
    TrackSamplingQueuedReset,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.selection import (
    TrackResetSelector,
    select_reset_track_by_course_id,
)

__all__ = (
    "TRACK_BASELINE_CACHE_LIMITS",
    "TRACK_SAMPLING_LIMITS",
    "SelectedTrack",
    "TrackBaselineCache",
    "TrackBaselineCacheLimits",
    "TrackSamplingDeficitLane",
    "TrackResetSelector",
    "TrackSamplingLimits",
    "TrackSamplingQueuedReset",
    "select_reset_track_by_course_id",
)
