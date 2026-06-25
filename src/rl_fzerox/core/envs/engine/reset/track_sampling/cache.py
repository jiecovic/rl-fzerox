# src/rl_fzerox/core/envs/engine/reset/track_sampling/cache.py
"""Per-worker baseline savestate cache for sampled tracks.

Loading baseline bytes through this cache avoids repeated disk reads while
keeping a bounded memory budget for long multi-course training runs.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.envs.engine.reset.track_sampling.models import (
    TRACK_BASELINE_CACHE_LIMITS,
)


class TrackBaselineCache:
    """Per-env cache of sampled track savestates.

    Subprocess workers keep their own cache, which avoids rereading the same
    multi-megabyte `.state` files on every episode reset. The cache is bounded
    by total bytes so large multi-track pools cannot grow worker memory
    without limit over long training runs.
    """

    def __init__(
        self,
        *,
        max_cached_state_bytes: int = TRACK_BASELINE_CACHE_LIMITS.max_cached_state_bytes,
    ) -> None:
        self._max_cached_state_bytes = max(0, int(max_cached_state_bytes))
        self._cached_state_bytes = 0
        self._states_by_path: OrderedDict[Path, bytes] = OrderedDict()

    def load_into_backend(self, backend: EmulatorBackend, path: Path) -> None:
        cache_path = _cache_path(path)
        state = self._states_by_path.get(cache_path)
        if state is None:
            state = path.read_bytes()
            self._remember_state(cache_path, state)
        else:
            self._states_by_path.move_to_end(cache_path)
        backend.load_baseline_bytes(state, source_path=path)

    def retain_paths(self, paths: Iterable[Path]) -> None:
        """Forget cached savestates that are no longer reset candidates."""

        retained_paths = {_cache_path(path) for path in paths}
        for cached_path in tuple(self._states_by_path):
            if cached_path in retained_paths:
                continue
            state = self._states_by_path.pop(cached_path)
            self._cached_state_bytes -= len(state)

    def _remember_state(self, path: Path, state: bytes) -> None:
        state_size = len(state)
        if self._max_cached_state_bytes <= 0 or state_size > self._max_cached_state_bytes:
            return
        while (
            self._states_by_path
            and self._cached_state_bytes + state_size > self._max_cached_state_bytes
        ):
            _, evicted_state = self._states_by_path.popitem(last=False)
            self._cached_state_bytes -= len(evicted_state)
        self._states_by_path[path] = state
        self._cached_state_bytes += state_size


def _cache_path(path: Path) -> Path:
    return path.expanduser().resolve()
