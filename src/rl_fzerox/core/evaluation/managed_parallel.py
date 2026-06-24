# src/rl_fzerox/core/evaluation/managed_parallel.py
"""Process-parallel execution for manager-owned evaluations."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Literal

from rl_fzerox.core.evaluation.artifacts import write_evaluation_result_files
from rl_fzerox.core.evaluation.engine_tuning import configure_evaluation_engine_tuning
from rl_fzerox.core.evaluation.executor import FZeroXSingleCourseEpisodeExecutor
from rl_fzerox.core.evaluation.models import (
    EvaluationAttemptResult,
    EvaluationCheckpointSnapshot,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRuntimeSpec,
)
from rl_fzerox.core.evaluation.runner import (
    EvaluationAttemptJob,
    EvaluationAttemptPlan,
    evaluation_attempt_from_course_result,
    evaluation_course_result_for_target,
)
from rl_fzerox.core.manager.models import ManagedEvaluation
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.inference import load_policy_runner
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.env import build_single_training_env


@dataclass(slots=True)
class _ParallelEvaluationWorkerContext:
    env: object
    executor: FZeroXSingleCourseEpisodeExecutor
    policy_path: Path
    policy_mode: EvaluationPolicyMode


_PARALLEL_WORKER_CONTEXT: _ParallelEvaluationWorkerContext | None = None


def run_parallel_managed_evaluation(
    evaluation: ManagedEvaluation,
    *,
    runtime_config: TrainAppConfig,
    run_paths: RunPaths,
    plan: EvaluationAttemptPlan,
    runtime: EvaluationRuntimeSpec,
    max_env_steps: int,
    should_cancel: Callable[[], bool] | None,
) -> EvaluationRunResult:
    """Execute a materialized attempt plan through isolated worker processes."""

    started_at_utc = _utc_now_text()
    attempts_by_index: dict[int, EvaluationAttemptResult] = {}

    if _cancel_requested(should_cancel):
        return _publish_managed_evaluation_result(
            evaluation,
            plan=plan,
            runtime=runtime,
            status="cancelled",
            started_at_utc=started_at_utc,
            attempts_by_index=attempts_by_index,
            closed_at_utc=_utc_now_text(),
        )

    _publish_managed_evaluation_result(
        evaluation,
        plan=plan,
        runtime=runtime,
        status="partial",
        started_at_utc=started_at_utc,
        attempts_by_index=attempts_by_index,
        closed_at_utc=None,
    )

    pool = ProcessPoolExecutor(
        max_workers=runtime.worker_count,
        mp_context=_evaluation_process_context(),
        initializer=_initialize_parallel_evaluation_worker,
        initargs=(
            runtime_config,
            evaluation.evaluation_dir,
            evaluation.checkpoint,
            evaluation.policy_mode,
            max_env_steps,
            run_paths.runtime_root,
        ),
    )
    pending: dict[Future[EvaluationAttemptResult], EvaluationAttemptJob] = {}
    remaining = iter(plan.jobs)
    cancelled = False

    def submit_next() -> bool:
        try:
            job = next(remaining)
        except StopIteration:
            return False
        pending[pool.submit(_run_parallel_evaluation_job, job)] = job
        return True

    try:
        for _ in range(runtime.worker_count):
            if not submit_next():
                break

        while pending:
            if _cancel_requested(should_cancel):
                cancelled = True
                break
            done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                job = pending.pop(future)
                attempts_by_index[job.attempt_index] = future.result()
                _publish_managed_evaluation_result(
                    evaluation,
                    plan=plan,
                    runtime=runtime,
                    status="partial",
                    started_at_utc=started_at_utc,
                    attempts_by_index=attempts_by_index,
                    closed_at_utc=None,
                )
                if not _cancel_requested(should_cancel):
                    submit_next()
                else:
                    cancelled = True
                    break
            if cancelled:
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    if cancelled or _cancel_requested(should_cancel):
        return _publish_managed_evaluation_result(
            evaluation,
            plan=plan,
            runtime=runtime,
            status="cancelled",
            started_at_utc=started_at_utc,
            attempts_by_index=attempts_by_index,
            closed_at_utc=_utc_now_text(),
        )

    return _publish_managed_evaluation_result(
        evaluation,
        plan=plan,
        runtime=runtime,
        status="completed",
        started_at_utc=started_at_utc,
        attempts_by_index=attempts_by_index,
        closed_at_utc=_utc_now_text(),
    )


def _publish_managed_evaluation_result(
    evaluation: ManagedEvaluation,
    *,
    plan: EvaluationAttemptPlan,
    runtime: EvaluationRuntimeSpec,
    status: Literal["completed", "cancelled", "partial"],
    started_at_utc: str,
    attempts_by_index: dict[int, EvaluationAttemptResult],
    closed_at_utc: str | None,
) -> EvaluationRunResult:
    result = EvaluationRunResult(
        spec=plan.spec,
        status=status,
        runtime=runtime,
        started_at_utc=started_at_utc,
        closed_at_utc=closed_at_utc,
        attempts=tuple(attempts_by_index[index] for index in sorted(attempts_by_index)),
    )
    write_evaluation_result_files(result, directory=evaluation.evaluation_dir)
    return result


def _initialize_parallel_evaluation_worker(
    config: TrainAppConfig,
    evaluation_dir: Path,
    checkpoint: EvaluationCheckpointSnapshot,
    policy_mode: EvaluationPolicyMode,
    max_env_steps: int,
    runtime_root: Path,
) -> None:
    global _PARALLEL_WORKER_CONTEXT

    policy_runner = load_policy_runner(
        evaluation_dir / "checkpoint_snapshot",
        artifact=checkpoint.artifact,
        device=config.train.device,
        algorithm=config.train.algorithm,
    )
    env = build_single_training_env(
        config,
        env_index=0,
        runtime_dir=runtime_root / f"eval_worker_{os.getpid()}",
    )
    configure_evaluation_engine_tuning(
        env,
        config,
        policy_path=Path(checkpoint.copied_policy_path),
    )
    _PARALLEL_WORKER_CONTEXT = _ParallelEvaluationWorkerContext(
        env=env,
        executor=FZeroXSingleCourseEpisodeExecutor(
            env=env,
            policy_runner=policy_runner,
            max_env_steps=max_env_steps,
        ),
        policy_path=Path(checkpoint.copied_policy_path),
        policy_mode=policy_mode,
    )
    atexit.register(_close_parallel_evaluation_worker)


def _run_parallel_evaluation_job(job: EvaluationAttemptJob) -> EvaluationAttemptResult:
    context = _parallel_worker_context()
    started_at_utc = _utc_now_text()
    course_result = evaluation_course_result_for_target(
        context.executor.run_course(
            job.target,
            policy_path=context.policy_path,
            policy_mode=context.policy_mode,
            seed=job.seed,
        ),
        target=job.target,
        seed=job.seed,
    )
    return evaluation_attempt_from_course_result(
        course_result,
        job=job,
        started_at_utc=started_at_utc,
        closed_at_utc=_utc_now_text(),
    )


def _parallel_worker_context() -> _ParallelEvaluationWorkerContext:
    if _PARALLEL_WORKER_CONTEXT is None:
        raise RuntimeError("parallel evaluation worker was not initialized")
    return _PARALLEL_WORKER_CONTEXT


def _close_parallel_evaluation_worker() -> None:
    global _PARALLEL_WORKER_CONTEXT

    context = _PARALLEL_WORKER_CONTEXT
    _PARALLEL_WORKER_CONTEXT = None
    if context is None:
        return
    close = getattr(context.env, "close", None)
    if callable(close):
        close()


def _evaluation_process_context() -> BaseContext:
    available_methods = mp.get_all_start_methods()
    if "forkserver" in available_methods:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


def _utc_now_text() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _cancel_requested(should_cancel: Callable[[], bool] | None) -> bool:
    return False if should_cancel is None else should_cancel()
