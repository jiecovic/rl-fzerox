# src/rl_fzerox/ui/watch/runtime/career_mode/loop/__init__.py
"""Career Mode watch loop facade.

The loop package owns worker orchestration around controller lifecycle,
recording, command draining, and snapshot publishing.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.runtime.career_mode.loop.entry import run_loaded_career_mode_loop

__all__ = [
    "run_loaded_career_mode_loop",
]
