# src/rl_fzerox/core/runtime_spec/track_registry/__init__.py
"""Track-registry expansion facade for runtime config loading.

Registry expansion turns compact course selections and registry references into
the concrete track and track-sampling entries consumed by runtime schemas.
"""

from __future__ import annotations

from .expand import expand_track_registry_metadata

__all__ = ["expand_track_registry_metadata"]
