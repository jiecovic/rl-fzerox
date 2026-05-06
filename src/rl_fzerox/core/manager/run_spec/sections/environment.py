# src/rl_fzerox/core/manager/run_spec/sections/environment.py
"""Environment/runtime section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, RendererName


class ManagedEnvironmentConfig(BaseModel):
    """Episode-limit and emulator-runtime knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    max_episode_steps: PositiveInt = 12_000
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    renderer: RendererName = DEFAULT_RENDERER
