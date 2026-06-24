# src/rl_fzerox/core/domain/courses/sampling.py
"""Track-sampling mode names shared by runtime and managed configs."""

from __future__ import annotations

from typing import Literal

type TrackSamplingMode = Literal[
    "equal",
    "step_balanced",
    "fixed_env",
    "deficit_budget",
]
type RuntimeTrackSamplingMode = TrackSamplingMode
type ManagedTrackSamplingMode = TrackSamplingMode
type DeficitBudgetDifficultyMetric = Literal[
    "completion_ema",
    "finish_ema",
    "mixed",
]
