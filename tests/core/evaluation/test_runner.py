# tests/core/evaluation/test_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rl_fzerox.core.evaluation import (
    CourseResultStatus,
    EvaluationCheckpointSnapshot,
    EvaluationCourseResult,
    EvaluationCourseTarget,
    EvaluationMode,
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationRuntimeSpec,
    EvaluationSpec,
    EvaluationTargetSpec,
    build_evaluation_attempt_plan,
    run_course_evaluation,
)


@dataclass
class _FakeEpisodeExecutor:
    statuses: tuple[CourseResultStatus, ...]
    calls: list[tuple[str, EvaluationPolicyMode, int]]

    def run_course(
        self,
        target: EvaluationCourseTarget,
        *,
        policy_mode: EvaluationPolicyMode,
        seed: int,
    ) -> EvaluationCourseResult:
        self.calls.append((target.target_id, policy_mode, seed))
        status = self.statuses[len(self.calls) - 1]
        return EvaluationCourseResult(
            course_id=target.course_id,
            status=status,
            race_time_ms=86_000 if status == "finished" else None,
            completion_ratio=1.0 if status == "finished" else 0.4,
            env_steps=3_000,
            episode_length_steps=3_000,
            episode_return=500.0 if status == "finished" else -6.0,
        )


class _Clock:
    def __init__(self) -> None:
        self._tick = 0

    def __call__(self) -> str:
        self._tick += 1
        return f"2026-06-22T10:00:{self._tick:02d}+00:00"


def test_course_runner_writes_progress_and_final_summary(
    tmp_path: Path,
) -> None:
    updates: list[EvaluationRunResult] = []
    executor = _FakeEpisodeExecutor(statuses=("finished", "crashed"), calls=[])
    spec = _evaluation_spec(tmp_path, repeats=2)
    target = EvaluationCourseTarget(
        target_id="mute-city-blue-falcon",
        course_id="mute_city",
        course_name="Mute City",
        cup_id="jack",
        difficulty="master",
        vehicle_id="blue_falcon",
        engine_setting_raw_value=90,
    )

    result = run_course_evaluation(
        spec,
        (target,),
        executor,
        runtime=EvaluationRuntimeSpec(device="cpu", worker_count=1),
        result_dir=tmp_path / "eval",
        on_update=updates.append,
        clock=_Clock(),
    )

    assert result.status == "completed"
    assert result.spec.total_planned_attempts == 2
    assert [attempt.status for attempt in result.attempts] == ["succeeded", "failed"]
    assert result.attempts[0].course_results[0].engine_setting_raw_value == 90
    assert executor.calls[0][1] == "deterministic"
    assert executor.calls[0][2] != executor.calls[1][2]
    assert [update.status for update in updates] == [
        "partial",
        "partial",
        "partial",
        "completed",
    ]
    assert len(updates[0].attempts) == 0

    payload = json.loads(
        (tmp_path / "eval" / "evaluation.summary.json").read_text(encoding="utf-8")
    )
    assert payload["result"]["status"] == "completed"
    assert payload["result"]["runtime"] == {"device": "cpu", "worker_count": 1}
    assert payload["metrics"]["overall"]["primary"]["attempt_count"] == 2
    assert payload["metrics"]["overall"]["primary"]["success_count"] == 1


def test_course_runner_rejects_empty_targets(tmp_path: Path) -> None:
    executor = _FakeEpisodeExecutor(statuses=(), calls=[])

    with pytest.raises(ValueError, match="at least one target"):
        run_course_evaluation(
            _evaluation_spec(tmp_path, repeats=1),
            (),
            executor,
            clock=_Clock(),
        )


def test_course_runner_repeats_target_set_in_rounds(tmp_path: Path) -> None:
    executor = _FakeEpisodeExecutor(
        statuses=("finished", "finished", "finished", "finished"),
        calls=[],
    )
    targets = (
        EvaluationCourseTarget(target_id="mute-city", course_id="mute_city"),
        EvaluationCourseTarget(target_id="silence", course_id="silence"),
    )

    result = run_course_evaluation(
        _evaluation_spec(tmp_path, repeats=2),
        targets,
        executor,
        clock=_Clock(),
    )

    assert result.spec.total_planned_attempts == 4
    assert [call[0] for call in executor.calls] == [
        "mute-city",
        "silence",
        "mute-city",
        "silence",
    ]


def test_attempt_plan_round_robins_baseline_variants_without_extra_attempts(
    tmp_path: Path,
) -> None:
    targets = tuple(
        EvaluationCourseTarget(
            target_id=f"mute-city-variant-{variant_index + 1}",
            course_id="mute_city",
            baseline_group_id="mute_city_gp",
            baseline_variant_index=variant_index,
            baseline_variant_count=10,
        )
        for variant_index in range(10)
    )

    plan = build_evaluation_attempt_plan(
        _evaluation_spec(
            tmp_path,
            repeats=10,
            mode="gp_course",
            baseline_variant_count=10,
        ),
        targets,
    )

    assert len(plan.jobs) == 10
    assert [job.target.target_id for job in plan.jobs] == [
        f"mute-city-variant-{variant_index + 1}" for variant_index in range(10)
    ]


def test_course_runner_attempt_plan_is_worker_partition_invariant(tmp_path: Path) -> None:
    targets = (
        EvaluationCourseTarget(target_id="mute-city", course_id="mute_city"),
        EvaluationCourseTarget(target_id="silence", course_id="silence"),
        EvaluationCourseTarget(target_id="sand-ocean", course_id="sand_ocean"),
    )

    plan = build_evaluation_attempt_plan(_evaluation_spec(tmp_path, repeats=3), targets)
    worker_one = tuple((job.attempt_index, job.target.target_id, job.seed) for job in plan.jobs)
    worker_four = tuple(
        sorted(
            (job.attempt_index, job.target.target_id, job.seed)
            for worker_index in range(4)
            for job in plan.jobs[worker_index::4]
        )
    )

    assert worker_four == worker_one


def test_course_runner_cancels_between_attempts(tmp_path: Path) -> None:
    updates: list[EvaluationRunResult] = []
    executor = _FakeEpisodeExecutor(statuses=("finished", "finished"), calls=[])
    cancel_checks = 0

    def should_cancel() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks > 1

    result = run_course_evaluation(
        _evaluation_spec(tmp_path, repeats=2),
        (EvaluationCourseTarget(target_id="mute-city", course_id="mute_city"),),
        executor,
        result_dir=tmp_path / "eval",
        on_update=updates.append,
        should_cancel=should_cancel,
        clock=_Clock(),
    )

    assert result.status == "cancelled"
    assert len(result.attempts) == 1
    assert [update.status for update in updates] == ["partial", "partial", "cancelled"]
    payload = json.loads(
        (tmp_path / "eval" / "evaluation.summary.json").read_text(encoding="utf-8")
    )
    assert payload["result"]["status"] == "cancelled"


def _evaluation_spec(
    tmp_path: Path,
    *,
    repeats: int,
    mode: EvaluationMode = "time_attack_course",
    baseline_variant_count: int = 1,
) -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id="eval-runner",
        seed=123,
        target=EvaluationTargetSpec(
            mode=mode,
            course_ids=("mute_city",),
            vehicle_ids=("blue_falcon",),
            repeats_per_target=repeats,
            baseline_variant_count=baseline_variant_count,
        ),
        checkpoint=EvaluationCheckpointSnapshot(
            source_run_id="run-a",
            source_run_name="Run A",
            artifact="latest",
            source_policy_path="/runs/run-a/checkpoints/latest/policy.zip",
            copied_policy_path=str(tmp_path / "checkpoints" / "latest" / "policy.zip"),
        ),
    )
