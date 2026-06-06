# src/rl_fzerox/core/career_mode/runner/policy.py
"""Policy handoff models for Career Mode races."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
    from rl_fzerox.core.training.inference import PolicyRunner


@dataclass(frozen=True, slots=True)
class CareerModePolicyControl:
    """Loaded trained policy selected for the current Career Mode race."""

    course_setup: ManagedSaveCourseSetup
    policy_run: ManagedRun
    runner: PolicyRunner

    @property
    def key(self) -> tuple[str, str]:
        return (
            self.course_setup.policy_run_id,
            self.course_setup.policy_artifact,
        )
