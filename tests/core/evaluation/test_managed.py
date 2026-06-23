# tests/core/evaluation/test_managed.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.evaluation.managed import (
    _evaluation_baseline_suite,
    _evaluation_materializer_input,
)
from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointSnapshot,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager import ManagedEvaluation, default_managed_run_config
from rl_fzerox.core.manager.training import build_managed_train_app_config


def test_shared_baseline_suite_ignores_repeat_count(tmp_path: Path) -> None:
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


def test_shared_baseline_suite_changes_with_seed_or_target(tmp_path: Path) -> None:
    base = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-001",
        seed=123,
        target=EvaluationTargetSpec(mode="gp_course", cup_ids=("joker",)),
    )
    different_seed = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-002",
        seed=456,
        target=EvaluationTargetSpec(mode="gp_course", cup_ids=("joker",)),
    )
    different_target = _managed_evaluation(
        tmp_path,
        evaluation_id="eval-003",
        seed=123,
        target=EvaluationTargetSpec(mode="gp_course", cup_ids=("queen",)),
    )

    base_suite = _suite_dir(base)

    assert _suite_dir(different_seed) != base_suite
    assert _suite_dir(different_target) != base_suite


def _suite_dir(evaluation: ManagedEvaluation) -> Path:
    config = build_managed_train_app_config(
        evaluation.config,
        run_id=evaluation.id,
        run_dir=evaluation.evaluation_dir / "runtime_projection",
    )
    materializer_input = _evaluation_materializer_input(config, seed=evaluation.seed)
    return _evaluation_baseline_suite(evaluation, materializer_input).run_paths.run_dir


def _managed_evaluation(
    tmp_path: Path,
    *,
    evaluation_id: str,
    target: EvaluationTargetSpec,
    seed: int = 123,
) -> ManagedEvaluation:
    return ManagedEvaluation(
        id=evaluation_id,
        name="Eval",
        status="created",
        evaluation_dir=tmp_path / "evaluations" / evaluation_id,
        source_run_id="run-001",
        source_artifact="latest",
        policy_mode="deterministic",
        seed=seed,
        target=target,
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
