# tests/core/evaluation/test_managed.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.evaluation.managed import _evaluation_baseline_suite
from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointSnapshot,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager import ManagedEvaluation, default_managed_run_config


def test_baseline_suite_uses_preset_version_key(tmp_path: Path) -> None:
    evaluation = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        preset_id="gp-master-blue-falcon",
        preset_version=3,
    )

    assert _suite_dir(evaluation) == (
        tmp_path / "evaluations" / "_baseline_suites" / "gp-master-blue-falcon-v3"
    )


def test_baseline_suite_is_shared_by_same_preset_version(tmp_path: Path) -> None:
    first = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        target=EvaluationTargetSpec(mode="gp_course", repeats_per_target=1),
    )
    second = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-002",
        target=EvaluationTargetSpec(mode="gp_course", repeats_per_target=10),
    )

    assert _suite_dir(second) == _suite_dir(first)


def test_baseline_suite_changes_with_preset_version(tmp_path: Path) -> None:
    base = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        preset_id="gp-master-blue-falcon",
        preset_version=1,
    )
    different_version = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-002",
        preset_id="gp-master-blue-falcon",
        preset_version=2,
    )

    assert _suite_dir(different_version) != _suite_dir(base)


def _suite_dir(evaluation: ManagedEvaluation) -> Path:
    return _evaluation_baseline_suite(evaluation).run_paths.run_dir


def _managed_evaluation(
    tmp_path: Path,
    *,
    evaluation_id: str,
    target: EvaluationTargetSpec | None = None,
    seed: int = 123,
    preset_id: str = "gp-master-blue-falcon",
    preset_version: int = 1,
) -> ManagedEvaluation:
    return ManagedEvaluation(
        id=evaluation_id,
        name="Eval",
        status="created",
        evaluation_dir=tmp_path / "evaluations" / evaluation_id,
        source_run_id="run-001",
        source_artifact="latest",
        preset_id=preset_id,
        preset_version=preset_version,
        policy_mode="deterministic",
        seed=seed,
        target=target or EvaluationTargetSpec(mode="gp_course"),
        config=default_managed_run_config(),
        checkpoint=EvaluationCheckpointSnapshot(
            source_run_id="run-001",
            source_run_name="Run 1",
            artifact="latest",
            source_policy_path="runs/run-001/checkpoints/latest/policy.zip",
            copied_policy_path="evaluations/eval/checkpoint_snapshot/policy.zip",
        ),
        created_at="2026-06-23T00:00:00+00:00",
        updated_at="2026-06-23T00:00:00+00:00",
    )
