# src/rl_fzerox/core/career_mode/execution/context.py
"""Execution context for one save-game unlock attempt.

The manager store assembles this immutable payload before the watch runtime
starts. Runtime code should consume the resolved values instead of re-querying
partial save-game setup state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.career_mode.course_setup import CourseSetupTarget
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    ManagedSaveUnlockTarget,
)


@dataclass(frozen=True, slots=True)
class SaveAttemptExecutionContext:
    """All manager-owned state needed before launching one unlock attempt."""

    save_game: ManagedSaveGame
    attempt: ManagedSaveAttempt
    target: ManagedSaveUnlockTarget
    course_setup_target: CourseSetupTarget
    course_setup: ManagedSaveCourseSetup
    cup_setup: ManagedSaveCupSetup
    policy_run: ManagedRun
    policy_artifact: Literal["latest", "best"]
    policy_path: Path
