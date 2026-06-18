# tests/ui/test_career_mode_loop_runtime.py
from __future__ import annotations

from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    policy_intro_wait_required,
)


def test_policy_intro_wait_required_until_training_target() -> None:
    assert policy_intro_wait_required(
        info={"race_intro_timer": 80},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 39},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 10},
        target_timer=39,
    )


def test_policy_intro_wait_required_skips_missing_or_disabled_target() -> None:
    assert not policy_intro_wait_required(
        info={"race_intro_timer": 80},
        target_timer=None,
    )
    assert not policy_intro_wait_required(
        info={},
        target_timer=39,
    )
    assert not policy_intro_wait_required(
        info={"race_intro_timer": "80"},
        target_timer=39,
    )
