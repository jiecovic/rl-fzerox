# src/rl_fzerox/core/evaluation/execution/publishing.py
"""Shared publication of evaluation progress and final results.

Evaluation runners own execution, cancellation, and worker scheduling. This
module owns the narrow side effect that every runner needs after state changes:
build the immutable run result, persist mirror artifacts when requested, and
notify live progress subscribers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.evaluation.models import (
    EvaluationAttemptResult,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationRuntimeSpec,
    EvaluationSpec,
)
from rl_fzerox.core.evaluation.reporting.artifacts import write_evaluation_result_files

type ProgressCallback = Callable[[EvaluationRunResult], None]


@dataclass(frozen=True, slots=True)
class EvaluationResultPublisher:
    """Writes and broadcasts one evaluation run's result snapshots."""

    spec: EvaluationSpec
    runtime: EvaluationRuntimeSpec
    started_at_utc: str
    result_dir: Path | None = None
    on_update: ProgressCallback | None = None

    def publish(
        self,
        *,
        status: EvaluationRunStatus,
        attempts: Iterable[EvaluationAttemptResult] = (),
        closed_at_utc: str | None = None,
    ) -> EvaluationRunResult:
        result = EvaluationRunResult(
            spec=self.spec,
            status=status,
            runtime=self.runtime,
            started_at_utc=self.started_at_utc,
            closed_at_utc=closed_at_utc,
            attempts=tuple(attempts),
        )
        if self.result_dir is not None:
            write_evaluation_result_files(result, directory=self.result_dir)
        if self.on_update is not None:
            self.on_update(result)
        return result


def attempts_in_index_order(
    attempts_by_index: Mapping[int, EvaluationAttemptResult],
) -> tuple[EvaluationAttemptResult, ...]:
    """Return parallel attempt results in the deterministic plan order."""

    return tuple(attempts_by_index[index] for index in sorted(attempts_by_index))
