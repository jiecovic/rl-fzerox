# src/rl_fzerox/core/evaluation/execution/__init__.py
"""Course-attempt execution primitives.

The execution package owns deterministic attempt planning, single-course
runner orchestration, concrete env/policy driving, and shared result publishing.
Manager-specific setup stays in ``evaluation.managed``.
"""

from rl_fzerox.core.evaluation.execution.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.execution.runner import (
    EvaluationAttemptJob,
    EvaluationAttemptPlan,
    SingleCourseEpisodeExecutor,
    build_evaluation_attempt_plan,
    run_course_evaluation,
)

__all__ = [
    "EvaluationAttemptJob",
    "EvaluationAttemptPlan",
    "FZeroXSingleCourseEpisodeExecutor",
    "SingleCourseEpisodeExecutor",
    "build_evaluation_attempt_plan",
    "run_course_evaluation",
]
