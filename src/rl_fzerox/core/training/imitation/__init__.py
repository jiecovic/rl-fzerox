# src/rl_fzerox/core/training/imitation/__init__.py
"""Shared teacher/student imitation primitives.

This package owns the cross-cutting seams needed by behavior cloning, DAgger,
and future distillation flows:

- observation-view planning for teacher and student
- canonical teacher/student action adaptation
- lightweight BC sample containers
"""

from __future__ import annotations

from .action_adapter import CanonicalControlIntent, TeacherStudentActionAdapter
from .behavior_cloning import BehaviorCloningBatch, BehaviorCloningSample
from .observations import (
    ObservationRenderPlan,
    ObservationViewSpec,
    observation_image_recipe,
    plan_observation_renders,
)

__all__ = [
    "BehaviorCloningBatch",
    "BehaviorCloningSample",
    "CanonicalControlIntent",
    "ObservationRenderPlan",
    "ObservationViewSpec",
    "TeacherStudentActionAdapter",
    "observation_image_recipe",
    "plan_observation_renders",
]
