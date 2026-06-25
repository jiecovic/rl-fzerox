# src/rl_fzerox/core/envs/engine/episode/__init__.py
"""Mutable per-episode engine state facade.

The env engine keeps episode-local progress, control, and reset metadata in one
owned dataclass. Runtime modules mutate that state explicitly through this
shared episode bookkeeping object.
"""

from rl_fzerox.core.envs.engine.episode.state import EngineEpisodeState

__all__ = ["EngineEpisodeState"]
