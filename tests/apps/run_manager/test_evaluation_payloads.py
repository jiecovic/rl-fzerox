# tests/apps/run_manager/test_evaluation_payloads.py
from __future__ import annotations

import json
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
        preset_id="time_attack_all_courses",
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
        id="time_attack_all_courses-v1",
        preset_id="time_attack_all_courses",
        preset_version=1,
        status="not_created",
        suite_dir=Path("local/evaluations/_baseline_suites/time_attack_all_courses-v1"),
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )

    payload = evaluation_payload(evaluation, baseline_suite=baseline_suite)
    checkpoint = payload["checkpoint"]

    assert isinstance(checkpoint, dict)
    assert checkpoint["source_mtime_ns"] == "1765000000000000123"
    assert payload["result_summary"] is None


def test_evaluation_payload_includes_result_summary(tmp_path: Path) -> None:
    result_path = tmp_path / "evaluation.summary.json"
    result_path.write_text(
        json.dumps(
            {
                "kind": "evaluation_summary",
                "schema_version": 1,
                "result": {
                    "status": "partial",
                    "started_at_utc": "2026-06-22T10:00:00+00:00",
                    "closed_at_utc": None,
                    "runtime": {"device": "cuda", "worker_count": 4},
                    "spec": {"total_planned_attempts": 2},
                    "attempts": [
                        {
                            "attempt_id": "attempt-0001",
                            "target_id": "mute_city",
                            "target_label": "Mute City",
                            "status": "succeeded",
                            "seed": 18_446_744_073_709_551_615,
                            "cup_id": "jack",
                            "difficulty": "master",
                            "vehicle_id": "blue_falcon",
                            "total_race_time_ms": 86_000,
                            "env_steps": 3_000,
                            "episode_return": 500.0,
                            "closed_at_utc": "2026-06-22T10:01:00+00:00",
                            "course_results": [
                                {
                                    "position": 1,
                                    "completion_ratio": 1.0,
                                }
                            ],
                        },
                        {
                            "attempt_id": "attempt-0002",
                            "target_id": "white_land",
                            "target_label": "White Land",
                            "status": "failed",
                            "seed": 123,
                            "cup_id": "king",
                            "difficulty": "master",
                            "vehicle_id": "blue_falcon",
                            "total_race_time_ms": 53_900,
                            "env_steps": 1_615,
                            "episode_return": 408.07,
                            "closed_at_utc": "2026-06-22T10:02:00+00:00",
                            "course_results": [
                                {
                                    "status": "crashed",
                                    "position": 1,
                                    "completion_ratio": 0.6,
                                }
                            ],
                        },
                    ],
                },
                "metrics": {
                    "overall": {
                        "key": "overall",
                        "label": "Overall",
                        "primary": {
                            "attempt_count": 1,
                            "success_count": 1,
                            "success_rate": 1.0,
                            "finish_count": 1,
                            "finish_rate": 1.0,
                            "completion_rate": 1.0,
                            "mean_finish_time_ms": 86_000.0,
                            "best_finish_time_ms": 86_000,
                            "mean_position": 1.0,
                            "best_position": 1,
                            "total_env_steps": 3_000,
                            "mean_episode_length_steps": 3_000.0,
                        },
                        "detail": {
                            "mean_episode_return": 500.0,
                            "best_episode_return": 500.0,
                            "average_speed": 720.0,
                        },
                    },
                    "courses": [],
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = ManagedEvaluation(
        id="eval-001",
        name="Eval 1",
        status="running",
        evaluation_dir=tmp_path,
        source_run_id="run-001",
        source_artifact="latest",
        preset_id="time_attack_all_courses",
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
        ),
        result_json_path=result_path,
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )
    baseline_suite = ManagedEvaluationBaselineSuite(
        id="time_attack_all_courses-v1",
        preset_id="time_attack_all_courses",
        preset_version=1,
        status="ready",
        suite_dir=tmp_path / "_baseline_suites" / "time_attack_all_courses-v1",
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )

    payload = evaluation_payload(evaluation, baseline_suite=baseline_suite)

    assert payload["progress"] == {
        "completed_attempts": 2,
        "total_attempts": 2,
        "result_status": "partial",
    }
    summary = payload["result_summary"]
    assert isinstance(summary, dict)
    assert summary["status"] == "partial"
    assert summary["runtime"] == {"device": "cuda", "worker_count": 4}
    assert summary["overall"] == {
        "key": "overall",
        "label": "Overall",
        "attempt_count": 1,
        "success_count": 1,
        "success_rate": 1.0,
        "finish_count": 1,
        "finish_rate": 1.0,
        "completion_rate": 1.0,
        "mean_finish_time_ms": 86000.0,
        "best_finish_time_ms": 86000.0,
        "mean_position": 1.0,
        "best_position": 1,
        "total_env_steps": 3000,
        "mean_episode_length_steps": 3000.0,
        "mean_episode_return": 500.0,
        "best_episode_return": 500.0,
        "average_speed": 720.0,
    }
    assert summary["attempts"][0]["seed"] == "18446744073709551615"
    assert summary["attempts"][0]["position"] == 1
    assert summary["attempts"][1]["position"] == 30
