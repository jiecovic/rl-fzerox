# src/rl_fzerox/ui/watch/runtime/worker.py
from __future__ import annotations

import sys

from rl_fzerox.ui.watch.runtime.live import worker as _worker
from rl_fzerox.ui.watch.runtime.live.worker import (
    _refresh_paused_cnn_activations,
    _sync_next_watch_reset_after_episode,
    _TimedWatchNotice,
    run_simulation_worker,
)

__all__ = [
    "_TimedWatchNotice",
    "_refresh_paused_cnn_activations",
    "_sync_next_watch_reset_after_episode",
    "run_simulation_worker",
]

# Preserve the historical module path as the same module object so tests,
# monkeypatches, and ad-hoc tools that patch worker globals still affect the
# implementation used by run_simulation_worker.
sys.modules[__name__] = _worker
