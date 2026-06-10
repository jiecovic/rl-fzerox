# src/rl_fzerox/apps/run_manager/worker/clock.py
from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def past_tense_command(command: str) -> str:
    return "stopped" if command == "stop" else f"{command}d"
