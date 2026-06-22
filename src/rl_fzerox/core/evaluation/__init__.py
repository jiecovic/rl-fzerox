# src/rl_fzerox/core/evaluation/__init__.py
"""Evaluation result contract, aggregate metrics, and local artifact writing."""

from rl_fzerox.core.evaluation.artifacts import (
    EvaluationArtifactPaths,
    write_evaluation_result_files,
)
from rl_fzerox.core.evaluation.executor import FZeroXSingleCourseEpisodeExecutor
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
    EvaluationCheckpointArtifact,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationCourseTarget,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationSpec,
    EvaluationTargetSpec,
)
from rl_fzerox.core.evaluation.runner import (
    SingleCourseEpisodeExecutor,
    run_headless_single_course_evaluation,
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
    "FZeroXSingleCourseEpisodeExecutor",
    "SingleCourseEpisodeExecutor",
    "aggregate_evaluation_metrics",
    "run_headless_single_course_evaluation",
    "snapshot_evaluation_checkpoint",
    "write_evaluation_result_files",
]
