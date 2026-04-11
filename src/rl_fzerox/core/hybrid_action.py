# src/rl_fzerox/core/hybrid_action.py
from __future__ import annotations

from typing import Final, Literal, TypeAlias

HybridActionBranchKey: TypeAlias = Literal["continuous", "discrete"]

HYBRID_CONTINUOUS_ACTION_KEY: Final[HybridActionBranchKey] = "continuous"
HYBRID_DISCRETE_ACTION_KEY: Final[HybridActionBranchKey] = "discrete"
