# src/rl_fzerox/core/evaluation/managed/__init__.py
"""Run-manager-owned evaluation entry points.

The package projects SQLite-owned evaluation records into train-like runtime
configs, then executes them through single-worker or process-parallel runners.
"""

from rl_fzerox.core.evaluation.managed.runner import (
    EvaluationBaselineSuite,
    run_managed_evaluation,
)

__all__ = [
    "EvaluationBaselineSuite",
    "run_managed_evaluation",
]
