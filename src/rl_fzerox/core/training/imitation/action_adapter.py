# src/rl_fzerox/core/training/imitation/action_adapter.py
"""Teacher/student action adaptation primitives for imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.envs.actions.base import ActionValue


@dataclass(frozen=True, slots=True)
class CanonicalControlIntent:
    """Project one action into shared continuous control semantics.

    Range conventions:

    - `steer`, `lean`, `pitch`: `[-1, 1]`
    - `gas`, `air_brake`, `boost`: `[0, 1]`
    """

    steer: float = 0.0
    gas: float = 0.0
    air_brake: float = 0.0
    boost: float = 0.0
    lean: float = 0.0
    pitch: float = 0.0

    def clamped(self) -> CanonicalControlIntent:
        """Return one bounded control intent using the canonical ranges."""

        return CanonicalControlIntent(
            steer=_clamp_signed(self.steer),
            gas=_clamp_unit(self.gas),
            air_brake=_clamp_unit(self.air_brake),
            boost=_clamp_unit(self.boost),
            lean=_clamp_signed(self.lean),
            pitch=_clamp_signed(self.pitch),
        )


class TeacherStudentActionAdapter(Protocol):
    """Translate teacher actions into student-supervision actions.

    Concrete implementations can:

    - decode teacher actions directly into canonical control intent
    - drop teacher-only controls
    - keep student-only controls neutral during imitation warm start
    """

    def teacher_action_to_canonical(self, action: ActionValue) -> CanonicalControlIntent:
        """Decode one teacher action into canonical control semantics."""
        ...

    def canonical_to_student_action(self, control: CanonicalControlIntent) -> ActionValue:
        """Encode canonical control semantics into one student action target."""
        ...


def _clamp_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _clamp_signed(value: float) -> float:
    return min(max(float(value), -1.0), 1.0)
