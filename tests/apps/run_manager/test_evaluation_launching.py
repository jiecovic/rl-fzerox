# tests/apps/run_manager/test_evaluation_launching.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

import pytest

from rl_fzerox.apps.run_manager.launching.evaluations import launch_evaluation_worker
from rl_fzerox.core.evaluation.models import EvaluationCheckpointSnapshot, EvaluationTargetSpec
from rl_fzerox.core.manager import ManagedEvaluation, ManagerStore, default_managed_run_config


def test_launch_evaluation_worker_passes_runtime_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = ManagedEvaluation(
        id="eval-001",
        name="Eval 1",
        status="created",
        evaluation_dir=tmp_path / "evaluations" / "eval-001",
        source_policy_kind="run",
        source_policy_id="run-001",
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
        created_at="2026-06-22T10:00:00+00:00",
        updated_at="2026-06-22T10:00:00+00:00",
    )
    captured: dict[str, object] = {}

    class _FakeStore(ManagerStore):
        def __init__(self) -> None:
            self.db_path = tmp_path / "manager" / "runs.db"

        def get_evaluation(self, evaluation_id: str) -> ManagedEvaluation | None:
            assert evaluation_id == evaluation.id
            return evaluation

        def mark_evaluation_running(self, evaluation_id: str) -> ManagedEvaluation:
            assert evaluation_id == evaluation.id
            return evaluation

        def mark_evaluation_failed(
            self,
            evaluation_id: str,
            *,
            error_message: str,
        ) -> ManagedEvaluation:
            raise AssertionError(f"unexpected evaluation failure: {evaluation_id} {error_message}")

    class _FakeProcess:
        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                return 0
            raise subprocess.TimeoutExpired(cmd="evaluation-worker", timeout=timeout or 0.0)

    def _fake_popen(
        command: list[str],
        *,
        cwd: Path,
        stdin: object,
        stdout: BinaryIO,
        stderr: object,
        start_new_session: bool,
    ) -> _FakeProcess:
        del cwd, stdin, stderr, start_new_session
        captured["command"] = command
        stdout.write(b"evaluation worker started\n")
        return _FakeProcess()

    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.evaluations.manager_evaluation_log_path",
        lambda evaluation_id: tmp_path / "logs" / f"{evaluation_id}.log",
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.evaluations.subprocess.Popen",
        _fake_popen,
    )
    monkeypatch.setattr(
        "rl_fzerox.apps.run_manager.launching.evaluations.reap_child_when_done",
        lambda process: None,
    )

    launched = launch_evaluation_worker(
        _FakeStore(),
        evaluation_id=evaluation.id,
        device="cpu",
        worker_count=4,
    )

    assert launched is evaluation
    assert captured["command"] == [
        sys.executable,
        "-m",
        "rl_fzerox.apps.evaluation_worker",
        "--db-path",
        str(tmp_path / "manager" / "runs.db"),
        "--evaluation-id",
        evaluation.id,
        "--device",
        "cpu",
        "--worker-count",
        "4",
    ]
