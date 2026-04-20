# src/rl_fzerox/core/envs/actions/hybrid/spaces.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator.arrays import ActionMask, NumpyArray
from rl_fzerox.core.domain.hybrid_action import (
    HYBRID_CONTINUOUS_ACTION_KEY,
    HYBRID_DISCRETE_ACTION_KEY,
)
from rl_fzerox.core.envs.actions.base import DiscreteActionDimension, build_flat_action_mask
from rl_fzerox.core.envs.actions.hybrid.layouts import HybridActionLayout


def hybrid_action_space(layout: HybridActionLayout) -> spaces.Dict:
    return spaces.Dict(
        {
            HYBRID_CONTINUOUS_ACTION_KEY: spaces.Box(
                low=np.full(layout.continuous_size, -1.0, dtype=np.float32),
                high=np.full(layout.continuous_size, 1.0, dtype=np.float32),
                dtype=np.float32,
            ),
            HYBRID_DISCRETE_ACTION_KEY: spaces.MultiDiscrete(
                np.array([dimension.size for dimension in layout.dimensions], dtype=np.int64)
            ),
        }
    )


def hybrid_idle_action(layout: HybridActionLayout) -> dict[str, NumpyArray]:
    return {
        HYBRID_CONTINUOUS_ACTION_KEY: np.zeros(layout.continuous_size, dtype=np.float32),
        HYBRID_DISCRETE_ACTION_KEY: np.zeros(layout.discrete_size, dtype=np.int64),
    }


def hybrid_action_mask(
    dimensions: tuple[DiscreteActionDimension, ...],
    *,
    base_overrides: dict[str, tuple[int, ...]] | None = None,
    stage_overrides: dict[str, tuple[int, ...]] | None = None,
    dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
) -> ActionMask:
    return build_flat_action_mask(
        dimensions,
        base_overrides=base_overrides,
        stage_overrides=stage_overrides,
        dynamic_overrides=dynamic_overrides,
    )
