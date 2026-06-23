# tests/apps/run_manager/test_evaluation_payloads.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.run_manager.api.payloads.evaluations import evaluation_payload
from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointSnapshot,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager import (
    ManagedEvaluation,
    ManagedEvaluationBaselineSuite,
    default_managed_run_config,
)


def test_evaluation_payload_serializes_source_mtime_ns_losslessly() -> None:
    evaluation = ManagedEvaluation(
        id="eval-001",
        name="Eval 1",
        status="created",
        evaluation_dir=Path("local/evaluations/eval-001"),
        source_run_id="run-001",
        source_artifact="latest",
        preset_id="time_attack_blue_falcon_all_courses",
        preset_version=1,
        policy_mode="deterministic",
        seed=123,
        target=EvaluationTargetSpec(mode="time_attack_course", repeats_per_target=1),
        config=default_managed_run_config(),
        checkpoint=EvaluationCheckpointSnapshot(
            source_run_id="run-001",
            source_run_name="Run 1",
            artifact="latest",
            source_policy_path="local/runs/run-001/checkpoints/latest/policy.zip",
            copied_policy_path="local/evaluations/eval-001/checkpoint_snapshot/policy.zip",
            source_mtime_ns=1_765_000_000_000_000_123,
        ),
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )
    baseline_suite = ManagedEvaluationBaselineSuite(
        id="time_attack_blue_falcon_all_courses-v1",
        preset_id="time_attack_blue_falcon_all_courses",
        preset_version=1,
        status="missing",
        suite_dir=Path("local/evaluations/_baseline_suites/time_attack_blue_falcon_all_courses-v1"),
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )

    payload = evaluation_payload(evaluation, baseline_suite=baseline_suite)
    checkpoint = payload["checkpoint"]

    assert isinstance(checkpoint, dict)
    assert checkpoint["source_mtime_ns"] == "1765000000000000123"
