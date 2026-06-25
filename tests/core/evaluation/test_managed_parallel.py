# tests/core/evaluation/test_managed_parallel.py
"""Tests for process-independent parallel evaluation scheduling."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field

import pytest

from rl_fzerox.core.evaluation.managed_parallel import _run_parallel_attempt_schedule
from rl_fzerox.core.evaluation.models import (
    EvaluationAttemptResult,
    EvaluationCourseTarget,
)
from rl_fzerox.core.evaluation.runner import EvaluationAttemptJob


def test_parallel_attempt_schedule_preserves_results_by_attempt_index() -> None:
    executor = _FakeAttemptExecutor(completion_order=[2, 1, 3])
    updates: list[tuple[int, ...]] = []

    result = _run_parallel_attempt_schedule(
        jobs=_jobs(1, 2, 3),
        worker_count=2,
        submit=executor.submit,
        wait_for_first_completed=executor.wait_for_next_completion,
        should_cancel=None,
        on_attempt_update=lambda attempts: updates.append(tuple(sorted(attempts))),
    )

    assert result.cancelled is False
    assert executor.submitted_indices == [1, 2, 3]
    assert updates == [(2,), (1, 2), (1, 2, 3)]
    attempt_ids = [
        result.attempts_by_index[index].attempt_id
        for index in sorted(result.attempts_by_index)
    ]
    assert attempt_ids == [
        "attempt-0001",
        "attempt-0002",
        "attempt-0003",
    ]


def test_parallel_attempt_schedule_stops_refilling_after_cancel() -> None:
    executor = _FakeAttemptExecutor(completion_order=[1])
    updates: list[tuple[int, ...]] = []

    result = _run_parallel_attempt_schedule(
        jobs=_jobs(1, 2, 3),
        worker_count=2,
        submit=executor.submit,
        wait_for_first_completed=executor.wait_for_next_completion,
        should_cancel=lambda: bool(updates),
        on_attempt_update=lambda attempts: updates.append(tuple(sorted(attempts))),
    )

    assert result.cancelled is True
    assert executor.submitted_indices == [1, 2]
    assert updates == [(1,)]
    assert sorted(result.attempts_by_index) == [1]


def test_parallel_attempt_schedule_propagates_worker_failures() -> None:
    executor = _FakeAttemptExecutor(completion_order=[1], failing_indices={1})
    updates: list[tuple[int, ...]] = []

    with pytest.raises(RuntimeError, match="attempt 1 failed"):
        _run_parallel_attempt_schedule(
            jobs=_jobs(1),
            worker_count=1,
            submit=executor.submit,
            wait_for_first_completed=executor.wait_for_next_completion,
            should_cancel=None,
            on_attempt_update=lambda attempts: updates.append(tuple(sorted(attempts))),
        )

    assert executor.submitted_indices == [1]
    assert updates == []


@dataclass(slots=True)
class _FakeAttemptExecutor:
    completion_order: list[int]
    failing_indices: set[int] = field(default_factory=set)
    submitted_indices: list[int] = field(default_factory=list)
    futures_by_index: dict[int, Future[EvaluationAttemptResult]] = field(
        default_factory=dict
    )

    def submit(self, job: EvaluationAttemptJob) -> Future[EvaluationAttemptResult]:
        future: Future[EvaluationAttemptResult] = Future()
        if job.attempt_index in self.failing_indices:
            future.set_exception(RuntimeError(f"attempt {job.attempt_index} failed"))
        else:
            future.set_result(_attempt_result(job.attempt_index))
        self.submitted_indices.append(job.attempt_index)
        self.futures_by_index[job.attempt_index] = future
        return future

    def wait_for_next_completion(
        self,
        pending: tuple[Future[EvaluationAttemptResult], ...],
    ) -> set[Future[EvaluationAttemptResult]]:
        attempt_index = self.completion_order.pop(0)
        future = self.futures_by_index[attempt_index]
        assert future in pending
        return {future}


def _jobs(*indices: int) -> tuple[EvaluationAttemptJob, ...]:
    return tuple(
        EvaluationAttemptJob(
            attempt_index=index,
            target=EvaluationCourseTarget(
                target_id=f"target-{index}",
                course_id=f"course-{index}",
            ),
            seed=100 + index,
        )
        for index in indices
    )


def _attempt_result(index: int) -> EvaluationAttemptResult:
    return EvaluationAttemptResult(
        attempt_id=f"attempt-{index:04d}",
        target_id=f"target-{index}",
        status="succeeded",
        seed=100 + index,
    )
