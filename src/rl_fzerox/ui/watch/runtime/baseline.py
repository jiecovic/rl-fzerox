# src/rl_fzerox/ui/watch/runtime/baseline.py
from __future__ import annotations

from pathlib import Path

from fzerox_emulator import Emulator


def _save_baseline_state(*, emulator: Emulator, baseline_state_path: Path | None) -> None:
    emulator.capture_current_as_baseline(baseline_state_path)
