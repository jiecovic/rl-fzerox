# src/rl_fzerox/core/evaluation/managed/__init__.py
"""Run-manager-owned evaluation execution entry points."""

from rl_fzerox.core.evaluation.managed.runner import (
    EvaluationBaselineSuite,
    run_managed_evaluation,
)

__all__ = [
    "EvaluationBaselineSuite",
    "run_managed_evaluation",
]
