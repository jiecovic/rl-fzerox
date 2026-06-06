# tests/ui/test_career_policy_race.py
from __future__ import annotations

from rl_fzerox.ui.watch.runtime.career_mode.policy_race import (
    career_policy_race_info,
)


def test_career_policy_race_info_drops_training_lifecycle_state() -> None:
    info = career_policy_race_info(
        {
            "game_mode": "gp_race",
            "termination_reason": "progress_stalled",
            "terminated": False,
            "truncated": True,
            "truncation_reason": "progress_stalled",
        }
    )

    assert info == {"game_mode": "gp_race"}


def test_career_policy_race_info_keeps_native_terminal_reason() -> None:
    info = career_policy_race_info(
        {
            "game_mode": "gp_race",
            "termination_reason": "finished",
            "truncated": False,
        }
    )

    assert info == {
        "game_mode": "gp_race",
        "termination_reason": "finished",
    }
