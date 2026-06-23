# src/rl_fzerox/core/domain/track_sampling.py
"""Track-sampling mode names shared by runtime and managed configs."""

from __future__ import annotations

from typing import Literal, TypeAlias

RuntimeTrackSamplingMode: TypeAlias = Literal[
    "random",
    "balanced",
    "step_balanced",
    # Accepted for older train manifests and saved configs. New manager UI flows
    # hide it until the policy for legacy cleanup changes.
    "adaptive_step_balanced",
    "fixed_env",
    "deficit_budget",
]
ManagedTrackSamplingMode: TypeAlias = Literal[
    "equal",
    "step_balanced",
    # Accepted for older managed configs. New manager UI flows hide it until the
    # policy for legacy cleanup changes.
    "adaptive_step_balanced",
    "fixed_env",
    "deficit_budget",
]
DeficitBudgetDifficultyMetric: TypeAlias = Literal[
    "completion_ema",
    "finish_ema",
    "mixed",
]
