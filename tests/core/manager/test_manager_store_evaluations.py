# tests/core/manager/test_manager_store_evaluations.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import policy_artifact_metadata_path


def _create_run_with_latest_checkpoint(store: ManagerStore, tmp_path: Path) -> ManagedRun:
    run = store.create_run(
        run_id="run-001",
        name="Policy Run",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "run-001",
        lineage_step_offset=10_000,
    )
    checkpoint_dir = run.run_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    checkpoint_dir.mkdir(parents=True)
    model_path = checkpoint_dir / "model.zip"
    policy_path = checkpoint_dir / "policy.zip"
    model_path.write_bytes(b"model")
    policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(policy_path).write_text(
        '{"num_timesteps": 123, "lineage_num_timesteps": 10123}\n',
        encoding="utf-8",
    )
    return run


def test_manager_store_creates_evaluation_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)

    evaluation = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        seed=42,
        target=EvaluationTargetSpec(mode="time_attack_course", repeats_per_target=3),
        config=run.config,
        evaluations_root=tmp_path / "evaluations",
    )

    assert "-eval-1-" in evaluation.id
    assert evaluation.status == "created"
    assert evaluation.checkpoint.source_run_id == run.id
    assert evaluation.checkpoint.local_num_timesteps == 123
    assert evaluation.checkpoint.lineage_num_timesteps == 10_123
    assert Path(evaluation.checkpoint.copied_policy_path).read_bytes() == b"policy"
    assert Path(evaluation.checkpoint.copied_model_path or "").read_bytes() == b"model"
    assert (evaluation.evaluation_dir / "evaluation.spec.json").is_file()
    assert (evaluation.evaluation_dir / "evaluation.config.json").is_file()

    reloaded = ManagerStore(store.db_path).list_evaluations()

    assert [candidate.id for candidate in reloaded] == [evaluation.id]
    assert reloaded[0].target.repeats_per_target == 3
    assert reloaded[0].config == run.config
    assert reloaded[0].checkpoint.copied_policy_path == evaluation.checkpoint.copied_policy_path


def test_manager_store_reuses_identical_created_evaluation_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    target = EvaluationTargetSpec(
        mode="gp_course",
        cup_ids=("joker",),
        difficulties=("master",),
        repeats_per_target=2,
    )

    first = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        seed=42,
        target=target,
        config=run.config,
        evaluations_root=tmp_path / "evaluations",
    )
    second = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        seed=42,
        target=target,
        config=run.config,
        evaluations_root=tmp_path / "evaluations",
    )

    assert second.id == first.id
    assert second.evaluation_dir == first.evaluation_dir
    assert [evaluation.id for evaluation in store.list_evaluations()] == [first.id]


def test_manager_store_deletes_created_evaluation_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    evaluation = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        seed=42,
        target=EvaluationTargetSpec(mode="time_attack_course", repeats_per_target=3),
        config=run.config,
        evaluations_root=tmp_path / "evaluations",
    )

    assert evaluation.evaluation_dir.is_dir()

    assert store.delete_evaluation(evaluation.id) is True

    assert store.list_evaluations() == ()
    assert not evaluation.evaluation_dir.exists()


def test_manager_store_updates_evaluation_lifecycle(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    evaluation = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        seed=42,
        target=EvaluationTargetSpec(mode="time_attack_course", repeats_per_target=3),
        config=run.config,
        evaluations_root=tmp_path / "evaluations",
    )

    running = store.mark_evaluation_running(evaluation.id)
    assert running.status == "running"
    assert running.started_at is not None
    assert running.result_json_path == evaluation.evaluation_dir / "evaluation.summary.json"

    completed = store.mark_evaluation_completed(evaluation.id)
    assert completed.status == "completed"
    assert completed.finished_at is not None
    assert completed.error_message is None


def test_manager_store_delete_missing_evaluation_returns_false(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    assert store.delete_evaluation("missing-eval") is False
