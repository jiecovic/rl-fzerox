# src/rl_fzerox/core/training/imitation/behavior_cloning.py
"""Small sample containers shared by imitation-learning trainers."""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions.base import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue


@dataclass(frozen=True, slots=True)
class BehaviorCloningSample:
    """One supervised student-target sample."""

    student_observation: ObservationValue
    teacher_action: ActionValue
    action_mask: ActionMask | None = None


@dataclass(frozen=True, slots=True)
class BehaviorCloningBatch:
    """One immutable batch of BC samples."""

    samples: tuple[BehaviorCloningSample, ...]

    def __len__(self) -> int:
        return len(self.samples)
