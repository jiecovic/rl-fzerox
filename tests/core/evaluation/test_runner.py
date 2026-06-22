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
    EvaluationPolicyMode,
    EvaluationRunResult,
    EvaluationSpec,
    EvaluationTargetSpec,
    run_headless_single_course_evaluation,
)


@dataclass
class _FakeEpisodeExecutor:
    statuses: tuple[CourseResultStatus, ...]
    calls: list[tuple[str, Path, EvaluationPolicyMode, int]]

    def run_course(
        self,
        target: EvaluationCourseTarget,
        *,
        policy_path: Path,
        policy_mode: EvaluationPolicyMode,
        seed: int,
    ) -> EvaluationCourseResult:
        self.calls.append((target.target_id, policy_path, policy_mode, seed))
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


def test_headless_single_course_runner_writes_progress_and_final_summary(
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

    result = run_headless_single_course_evaluation(
        spec,
        (target,),
        executor,
        result_dir=tmp_path / "eval",
        on_update=updates.append,
        clock=_Clock(),
    )

    assert result.status == "completed"
    assert result.spec.total_planned_attempts == 2
    assert [attempt.status for attempt in result.attempts] == ["succeeded", "failed"]
    assert result.attempts[0].course_results[0].engine_setting_raw_value == 90
    assert executor.calls[0][1] == tmp_path / "checkpoints" / "latest" / "policy.zip"
    assert executor.calls[0][2] == "deterministic"
    assert executor.calls[0][3] != executor.calls[1][3]
    assert [update.status for update in updates] == ["partial", "partial", "completed"]

    payload = json.loads(
        (tmp_path / "eval" / "evaluation.summary.json").read_text(encoding="utf-8")
    )
    assert payload["result"]["status"] == "completed"
    assert payload["metrics"]["overall"]["primary"]["attempt_count"] == 2
    assert payload["metrics"]["overall"]["primary"]["success_count"] == 1


def test_headless_single_course_runner_rejects_empty_targets(tmp_path: Path) -> None:
    executor = _FakeEpisodeExecutor(statuses=(), calls=[])

    with pytest.raises(ValueError, match="at least one target"):
        run_headless_single_course_evaluation(
            _evaluation_spec(tmp_path, repeats=1),
            (),
            executor,
            clock=_Clock(),
        )


def _evaluation_spec(tmp_path: Path, *, repeats: int) -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id="eval-runner",
        seed=123,
        target=EvaluationTargetSpec(
            mode="time_attack",
            course_ids=("mute_city",),
            vehicle_ids=("blue_falcon",),
            repeats_per_target=repeats,
        ),
        checkpoint=EvaluationCheckpointSnapshot(
            source_run_id="run-a",
            source_run_name="Run A",
            artifact="latest",
            source_policy_path="/runs/run-a/checkpoints/latest/policy.zip",
            copied_policy_path=str(tmp_path / "checkpoints" / "latest" / "policy.zip"),
        ),
    )
