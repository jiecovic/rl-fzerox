# src/rl_fzerox/core/runtime_spec/schema/emulator.py
"""Runtime emulator boot-path schema.

This module owns the libretro core/content paths and optional runtime state
paths needed before env or watch code can create an emulator instance.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, FilePath

from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, RendererName


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None
    renderer: RendererName = DEFAULT_RENDERER
