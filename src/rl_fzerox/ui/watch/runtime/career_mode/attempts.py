# src/rl_fzerox/ui/watch/runtime/career_mode/attempts.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.manager import ManagerStore


@dataclass(frozen=True, slots=True)
class RunnerCloseReasons:
    """Stable failure-reason text written when the standalone runner exits."""

    failed: str = "career mode runner failed"
    closed: str = "career mode runner closed"


RUNNER_CLOSE_REASONS = RunnerCloseReasons()


def fail_running_attempts(
    store: ManagerStore,
    *,
    save_game_id: str,
    failure_reason: str,
) -> None:
    """Preserve running Career Mode attempts as failed history rows."""

    store.fail_running_save_attempts(
        save_game_id=save_game_id,
        failure_reason=failure_reason,
    )
    save_game = store.get_save_game(save_game_id)
    if save_game is not None and save_game.status == "running":
        store.update_save_game_status(save_game_id=save_game_id, status="paused")
