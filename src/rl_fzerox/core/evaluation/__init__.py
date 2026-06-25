# src/rl_fzerox/core/evaluation/__init__.py
"""Evaluation result contract, aggregate metrics, and local artifact writing."""

from rl_fzerox.core.evaluation.execution import (
    EvaluationAttemptJob,
    EvaluationAttemptPlan,
    FZeroXSingleCourseEpisodeExecutor,
    SingleCourseEpisodeExecutor,
    build_evaluation_attempt_plan,
    run_course_evaluation,
)
from rl_fzerox.core.evaluation.models import (
    AttemptStatus,
    CourseResultStatus,
    EvaluationAttemptResult,
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationCourseTarget,
    EvaluationDevice,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationRuntimeSpec,
    EvaluationSpec,
    EvaluationTargetSpec,
)
from rl_fzerox.core.evaluation.reporting import (
    EvaluationArtifactPaths,
    EvaluationDetailMetrics,
    EvaluationMetricGroup,
    EvaluationMetrics,
    EvaluationPrimaryMetrics,
    aggregate_evaluation_metrics,
    write_evaluation_result_files,
)
from rl_fzerox.core.evaluation.snapshots import (
    EvaluationCheckpointSource,
    snapshot_evaluation_checkpoint,
)

__all__ = [
    "AttemptStatus",
    "CourseResultStatus",
    "EvaluationArtifactPaths",
    "EvaluationAttemptResult",
    "EvaluationCheckpointArtifact",
    "EvaluationCheckpointSource",
    "EvaluationCheckpointSnapshot",
    "EvaluationCourseResult",
    "EvaluationCourseTarget",
    "EvaluationDevice",
    "EvaluationDetailMetrics",
    "EvaluationMetricGroup",
    "EvaluationMetrics",
    "EvaluationMode",
    "EvaluationPolicyMode",
    "EvaluationPrimaryMetrics",
    "EvaluationRunResult",
    "EvaluationRunStatus",
    "EvaluationRuntimeSpec",
    "EvaluationSpec",
    "EvaluationTargetSpec",
    "EvaluationAttemptJob",
    "EvaluationAttemptPlan",
    "FZeroXSingleCourseEpisodeExecutor",
    "SingleCourseEpisodeExecutor",
    "aggregate_evaluation_metrics",
    "build_evaluation_attempt_plan",
    "run_course_evaluation",
    "snapshot_evaluation_checkpoint",
    "write_evaluation_result_files",
]
