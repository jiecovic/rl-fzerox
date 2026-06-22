# src/rl_fzerox/core/evaluation/__init__.py
"""Evaluation result contract, aggregate metrics, and local artifact writing."""

from rl_fzerox.core.evaluation.artifacts import (
    EvaluationArtifactPaths,
    write_evaluation_result_files,
)
from rl_fzerox.core.evaluation.metrics import (
    EvaluationDetailMetrics,
    EvaluationMetricGroup,
    EvaluationMetrics,
    EvaluationPrimaryMetrics,
    aggregate_evaluation_metrics,
)
from rl_fzerox.core.evaluation.models import (
    AttemptStatus,
    CourseResultStatus,
    EvaluationAttemptResult,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationSpec,
    EvaluationTargetSpec,
)

__all__ = [
    "AttemptStatus",
    "CourseResultStatus",
    "EvaluationArtifactPaths",
    "EvaluationAttemptResult",
    "EvaluationCheckpointSnapshot",
    "EvaluationCourseResult",
    "EvaluationDetailMetrics",
    "EvaluationMetricGroup",
    "EvaluationMetrics",
    "EvaluationMode",
    "EvaluationPolicyMode",
    "EvaluationPrimaryMetrics",
    "EvaluationRunResult",
    "EvaluationRunStatus",
    "EvaluationSpec",
    "EvaluationTargetSpec",
    "aggregate_evaluation_metrics",
    "write_evaluation_result_files",
]
