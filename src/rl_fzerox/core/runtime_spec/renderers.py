# src/rl_fzerox/core/runtime_spec/renderers.py
"""Renderer names accepted by runtime config and app launchers.

The values here are wire-level config names. Env and watch code can rely on
this small catalog instead of duplicating renderer string literals.
"""

from __future__ import annotations

from typing import Final, Literal

type RendererName = Literal["angrylion", "gliden64"]

DEFAULT_RENDERER: Final[RendererName] = "gliden64"
KNOWN_RENDERERS: Final[tuple[RendererName, ...]] = ("angrylion", "gliden64")
