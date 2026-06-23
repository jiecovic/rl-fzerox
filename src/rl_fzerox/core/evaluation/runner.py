# src/rl_fzerox/core/evaluation/runner.py
"""Evaluation orchestration for deterministic course-attempt suites."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeAlias

from rl_fzerox.core.evaluation.artifacts import write_evaluation_result_files
from rl_fzerox.core.evaluation.models import (
    AttemptStatus,
    CourseResultStatus,
    EvaluationAttemptResult,
    EvaluationCourseResult,
    EvaluationCourseTarget,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRuntimeSpec,
    EvaluationSpec,
)
from rl_fzerox.core.seed import derive_seed

ProgressCallback: TypeAlias = Callable[[EvaluationRunResult], None]
CancelCheck: TypeAlias = Callable[[], bool]
Clock: TypeAlias = Callable[[], str]


@dataclass(frozen=True, slots=True)
class _EvaluationRunnerSeedDomains:
    """Domain ids for deterministic per-attempt substreams."""

    attempt: int = int.from_bytes(b"evalatt1", "big")


_SEED_DOMAINS = _EvaluationRunnerSeedDomains()


@dataclass(frozen=True, slots=True)
class EvaluationAttemptJob:
    """One deterministic course-run job in a materialized evaluation plan."""

    attempt_index: int
    target: EvaluationCourseTarget
    seed: int


@dataclass(frozen=True, slots=True)
class EvaluationAttemptPlan:
    """Stable attempt order and seed assignment for one evaluation run."""

    spec: EvaluationSpec
    jobs: tuple[EvaluationAttemptJob, ...]


class SingleCourseEpisodeExecutor(Protocol):
    """Runs one already-expanded course target with one frozen policy."""

    def run_course(
        self,
        target: EvaluationCourseTarget,
        *,
        policy_path: Path,
        policy_mode: EvaluationPolicyMode,
        seed: int,
    ) -> EvaluationCourseResult:
        """Drive one target episode and return the terminal course result."""
        ...


def _utc_now_text() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def run_course_evaluation(
    spec: EvaluationSpec,
    targets: Iterable[EvaluationCourseTarget],
    executor: SingleCourseEpisodeExecutor,
    *,
    runtime: EvaluationRuntimeSpec | None = None,
    result_dir: Path | None = None,
    on_update: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    clock: Clock = _utc_now_text,
) -> EvaluationRunResult:
    """Run a deterministic single-course evaluation suite.

    The concrete emulator/policy integration is intentionally behind
    ``SingleCourseEpisodeExecutor``. That keeps lifecycle, checkpoint identity,
    summary writing, and progress updates independent from the runtime that
    actually advances F-Zero X.
    """

    plan = build_evaluation_attempt_plan(spec, targets)
    run_spec = plan.spec
    run_runtime = runtime or EvaluationRuntimeSpec()
    started_at_utc = clock()
    attempts: list[EvaluationAttemptResult] = []
    policy_path = Path(run_spec.checkpoint.copied_policy_path)

    for job in plan.jobs:
        if _cancel_requested(should_cancel):
            return _publish_cancelled_result(
                run_spec,
                runtime=run_runtime,
                started_at_utc=started_at_utc,
                attempts=attempts,
                result_dir=result_dir,
                on_update=on_update,
                clock=clock,
            )
        attempt_started_at_utc = clock()
        course_result = evaluation_course_result_for_target(
            executor.run_course(
                job.target,
                policy_path=policy_path,
                policy_mode=run_spec.policy_mode,
                seed=job.seed,
            ),
            target=job.target,
            seed=job.seed,
        )
        attempts.append(
            evaluation_attempt_from_course_result(
                course_result,
                job=job,
                started_at_utc=attempt_started_at_utc,
                closed_at_utc=clock(),
            )
        )
        _publish_result_update(
            EvaluationRunResult(
                spec=run_spec,
                status="partial",
                runtime=run_runtime,
                started_at_utc=started_at_utc,
                attempts=tuple(attempts),
            ),
            result_dir=result_dir,
            on_update=on_update,
        )
        if _cancel_requested(should_cancel):
            return _publish_cancelled_result(
                run_spec,
                runtime=run_runtime,
                started_at_utc=started_at_utc,
                attempts=attempts,
                result_dir=result_dir,
                on_update=on_update,
                clock=clock,
            )

    result = EvaluationRunResult(
        spec=run_spec,
        status="completed",
        runtime=run_runtime,
        started_at_utc=started_at_utc,
        closed_at_utc=clock(),
        attempts=tuple(attempts),
    )
    _publish_result_update(result, result_dir=result_dir, on_update=on_update)
    return result


def _cancel_requested(should_cancel: CancelCheck | None) -> bool:
    return False if should_cancel is None else should_cancel()


def build_evaluation_attempt_plan(
    spec: EvaluationSpec,
    targets: Iterable[EvaluationCourseTarget],
) -> EvaluationAttemptPlan:
    """Materialize target order and per-attempt seeds before execution starts."""

    if spec.target.mode not in ("time_attack_course", "gp_course"):
        raise ValueError(f"course evaluation does not support mode={spec.target.mode!r}")

    expanded_targets = _expand_targets(targets, repeats=spec.target.repeats_per_target)
    run_spec = replace(
        spec,
        total_planned_attempts=spec.total_planned_attempts or len(expanded_targets),
    )
    jobs = tuple(
        EvaluationAttemptJob(
            attempt_index=index,
            target=target,
            seed=_attempt_seed(run_spec.seed, index),
        )
        for index, target in enumerate(expanded_targets, start=1)
    )
    return EvaluationAttemptPlan(spec=run_spec, jobs=jobs)


def _publish_cancelled_result(
    spec: EvaluationSpec,
    *,
    runtime: EvaluationRuntimeSpec,
    started_at_utc: str,
    attempts: list[EvaluationAttemptResult],
    result_dir: Path | None,
    on_update: ProgressCallback | None,
    clock: Clock,
) -> EvaluationRunResult:
    result = EvaluationRunResult(
        spec=spec,
        status="cancelled",
        runtime=runtime,
        started_at_utc=started_at_utc,
        closed_at_utc=clock(),
        attempts=tuple(attempts),
    )
    _publish_result_update(result, result_dir=result_dir, on_update=on_update)
    return result


def _expand_targets(
    targets: Iterable[EvaluationCourseTarget],
    *,
    repeats: int,
) -> tuple[EvaluationCourseTarget, ...]:
    if repeats < 1:
        raise ValueError(f"repeats_per_target must be at least 1, got {repeats}")
    concrete_targets = tuple(targets)
    if not concrete_targets:
        raise ValueError("single-course evaluation requires at least one target")
    return tuple(target for _ in range(repeats) for target in concrete_targets)


def _attempt_seed(master_seed: int, attempt_index: int) -> int:
    seed = derive_seed(master_seed, _SEED_DOMAINS.attempt, attempt_index)
    if seed is None:
        raise RuntimeError("evaluation attempt seed derivation unexpectedly returned None")
    return seed


def evaluation_course_result_for_target(
    result: EvaluationCourseResult,
    *,
    target: EvaluationCourseTarget,
    seed: int,
) -> EvaluationCourseResult:
    return replace(
        result,
        course_id=result.course_id or target.course_id,
        course_name=result.course_name or target.course_name,
        cup_id=result.cup_id or target.cup_id,
        difficulty=result.difficulty or target.difficulty,
        vehicle_id=result.vehicle_id or target.vehicle_id,
        seed=result.seed if result.seed is not None else seed,
        engine_setting_raw_value=(
            result.engine_setting_raw_value
            if result.engine_setting_raw_value is not None
            else target.engine_setting_raw_value
        ),
    )


def evaluation_attempt_from_course_result(
    course_result: EvaluationCourseResult,
    *,
    job: EvaluationAttemptJob,
    started_at_utc: str,
    closed_at_utc: str,
) -> EvaluationAttemptResult:
    return EvaluationAttemptResult(
        attempt_id=f"attempt-{job.attempt_index:04d}",
        target_id=job.target.target_id,
        target_label=job.target.course_name or job.target.course_id,
        status=_attempt_status(course_result.status),
        cup_id=job.target.cup_id,
        difficulty=job.target.difficulty,
        vehicle_id=job.target.vehicle_id,
        seed=job.seed,
        started_at_utc=started_at_utc,
        closed_at_utc=closed_at_utc,
        total_race_time_ms=course_result.race_time_ms,
        env_steps=course_result.env_steps,
        episode_length_steps=course_result.episode_length_steps,
        episode_return=course_result.episode_return,
        course_results=(course_result,),
    )


def _attempt_status(status: CourseResultStatus) -> AttemptStatus:
    if status == "finished":
        return "succeeded"
    if status == "truncated":
        return "partial"
    return "failed"


def _publish_result_update(
    result: EvaluationRunResult,
    *,
    result_dir: Path | None,
    on_update: ProgressCallback | None,
) -> None:
    if result_dir is not None:
        write_evaluation_result_files(result, directory=result_dir)
    if on_update is not None:
        on_update(result)
