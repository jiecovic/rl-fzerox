# src/rl_fzerox/core/runtime_spec/renderers.py
from __future__ import annotations

from typing import Final, Literal

type RendererName = Literal["angrylion", "gliden64"]

DEFAULT_RENDERER: Final[RendererName] = "gliden64"
KNOWN_RENDERERS: Final[tuple[RendererName, ...]] = ("angrylion", "gliden64")
