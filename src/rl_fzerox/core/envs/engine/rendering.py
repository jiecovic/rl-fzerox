# src/rl_fzerox/core/envs/engine/rendering.py
"""Renderer resolution for emulator-backed env components."""

from __future__ import annotations

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, KNOWN_RENDERERS, RendererName

_RENDERERS_BY_NAME: dict[str, RendererName] = {name: name for name in KNOWN_RENDERERS}


def backend_renderer(backend: EmulatorBackend) -> RendererName:
    """Return the backend renderer name used for renderer-sized observations."""

    renderer = getattr(backend, "renderer", DEFAULT_RENDERER)
    return _RENDERERS_BY_NAME.get(str(renderer), DEFAULT_RENDERER)
