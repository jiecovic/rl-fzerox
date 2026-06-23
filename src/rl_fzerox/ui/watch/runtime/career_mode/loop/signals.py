# src/rl_fzerox/ui/watch/runtime/career_mode/loop/signals.py
from __future__ import annotations


class CareerModeWorkerQuit(Exception):
    """Internal signal used to unwind the Career Mode worker loop."""
