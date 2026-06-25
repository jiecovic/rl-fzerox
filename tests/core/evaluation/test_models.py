# tests/core/evaluation/test_models.py
from __future__ import annotations

import pytest

from rl_fzerox.core.evaluation import (
    EvaluationCheckpointSnapshot,
    EvaluationCourseTarget,
    EvaluationRuntimeSpec,
    EvaluationSpec,
    EvaluationTargetSpec,
)


def test_evaluation_target_spec_rejects_invalid_repeats() -> None:
    with pytest.raises(ValueError, match="repeats_per_target must be at least 1"):
        EvaluationTargetSpec(mode="gp_course", repeats_per_target=0)


def test_evaluation_target_spec_rejects_empty_filter_ids() -> None:
    with pytest.raises(ValueError, match=r"course_ids\[0\] must be a non-empty string"):
        EvaluationTargetSpec(mode="time_attack_course", course_ids=("",))


def test_evaluation_course_target_rejects_missing_identity() -> None:
    with pytest.raises(ValueError, match="target_id must be a non-empty string"):
        EvaluationCourseTarget(target_id="", course_id="mute_city")

    with pytest.raises(ValueError, match="course_id must be a non-empty string"):
        EvaluationCourseTarget(target_id="mute-city", course_id="")


def test_evaluation_checkpoint_snapshot_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="source_policy_path must be a non-empty string"):
        _checkpoint(source_policy_path="")

    with pytest.raises(ValueError, match="copied_policy_path must be a non-empty string"):
        _checkpoint(copied_policy_path=" ")


def test_evaluation_checkpoint_snapshot_rejects_negative_metadata() -> None:
    with pytest.raises(ValueError, match="local_num_timesteps must be at least 0"):
        _checkpoint(local_num_timesteps=-1)


def test_evaluation_spec_rejects_invalid_identity_and_counts() -> None:
    with pytest.raises(ValueError, match="evaluation_id must be a non-empty string"):
        _spec(evaluation_id="")

    with pytest.raises(ValueError, match="seed must be at least 0"):
        _spec(seed=-1)

    with pytest.raises(ValueError, match="total_planned_attempts must be at least 1"):
        _spec(total_planned_attempts=0)


def test_evaluation_runtime_spec_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="worker_count must be at least 1"):
        EvaluationRuntimeSpec(worker_count=0)


def _spec(
    *,
    evaluation_id: str = "eval-models",
    seed: int = 123,
    total_planned_attempts: int | None = None,
) -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id=evaluation_id,
        seed=seed,
        target=EvaluationTargetSpec(mode="gp_course"),
        checkpoint=_checkpoint(),
        total_planned_attempts=total_planned_attempts,
    )


def _checkpoint(
    *,
    source_policy_path: str = "/runs/run-a/checkpoints/latest/policy.zip",
    copied_policy_path: str = "/evals/eval-a/checkpoint_snapshot/policy.zip",
    local_num_timesteps: int | None = None,
) -> EvaluationCheckpointSnapshot:
    return EvaluationCheckpointSnapshot(
        source_run_id="run-a",
        source_run_name="Run A",
        artifact="latest",
        source_policy_path=source_policy_path,
        copied_policy_path=copied_policy_path,
        local_num_timesteps=local_num_timesteps,
    )
