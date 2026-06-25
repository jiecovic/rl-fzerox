# src/rl_fzerox/core/evaluation/reporting/__init__.py
"""Evaluation reporting facade.

This package turns immutable evaluation results into aggregate metric groups and
writes the JSON/Markdown artifact mirror consumed by the run manager UI and
local debugging workflows.
"""

from rl_fzerox.core.evaluation.reporting.artifacts import (
    EvaluationArtifactPaths,
    write_evaluation_result_files,
)
from rl_fzerox.core.evaluation.reporting.metrics import (
    EvaluationDetailMetrics,
    EvaluationMetricGroup,
    EvaluationMetrics,
    EvaluationPrimaryMetrics,
    aggregate_evaluation_metrics,
)

__all__ = [
    "EvaluationArtifactPaths",
    "EvaluationDetailMetrics",
    "EvaluationMetricGroup",
    "EvaluationMetrics",
    "EvaluationPrimaryMetrics",
    "aggregate_evaluation_metrics",
    "write_evaluation_result_files",
]
