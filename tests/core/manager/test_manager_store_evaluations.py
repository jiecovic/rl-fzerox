# tests/core/manager/test_manager_store_evaluations.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import policy_artifact_metadata_path

TIME_ATTACK_PRESET_ID = "time_attack_all_courses"
GP_PRESET_ID = "gp_course_master_all_courses"


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
        preset_id=TIME_ATTACK_PRESET_ID,
        device="cpu",
        evaluations_root=tmp_path / "evaluations",
    )

    assert "-eval-1-" in evaluation.id
    assert evaluation.status == "created"
    assert evaluation.source_artifact == "latest"
    assert evaluation.checkpoint.source_run_id == run.id
    assert evaluation.checkpoint.local_num_timesteps == 123
    assert evaluation.checkpoint.lineage_num_timesteps == 10_123
    assert Path(evaluation.checkpoint.copied_policy_path).read_bytes() == b"policy"
    assert Path(evaluation.checkpoint.copied_model_path or "").read_bytes() == b"model"
    assert evaluation.preset_id == TIME_ATTACK_PRESET_ID
    assert evaluation.preset_version == 1
    assert (evaluation.evaluation_dir / "evaluation.spec.json").is_file()
    assert (evaluation.evaluation_dir / "evaluation.config.json").is_file()

    reloaded = ManagerStore(store.db_path).list_evaluations()

    assert [candidate.id for candidate in reloaded] == [evaluation.id]
    assert reloaded[0].target.mode == "time_attack_course"
    assert reloaded[0].target.repeats_per_target == 10
    assert reloaded[0].config.environment.renderer == "gliden64"
    assert reloaded[0].config.train.device == "cpu"
    assert reloaded[0].target.vehicle_ids == run.config.vehicle.selected_vehicle_ids
    assert reloaded[0].checkpoint.copied_policy_path == evaluation.checkpoint.copied_policy_path


def test_manager_store_reuses_identical_created_evaluation_snapshot(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    first = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        preset_id=GP_PRESET_ID,
        evaluations_root=tmp_path / "evaluations",
    )
    second = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        preset_id=GP_PRESET_ID,
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
        preset_id=TIME_ATTACK_PRESET_ID,
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
        preset_id=TIME_ATTACK_PRESET_ID,
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


def test_manager_store_requests_evaluation_cancel(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    evaluation = store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        preset_id=TIME_ATTACK_PRESET_ID,
        evaluations_root=tmp_path / "evaluations",
    )
    store.mark_evaluation_running(evaluation.id)

    cancelled = store.request_evaluation_cancel(evaluation.id)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert store.evaluation_cancel_request_path(evaluation.id).is_file()


def test_manager_store_delete_missing_evaluation_returns_false(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    assert store.delete_evaluation("missing-eval") is False


def test_manager_store_lists_default_evaluation_presets_and_suites(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    presets = store.list_evaluation_presets()
    suites = store.list_evaluation_baseline_suites()

    assert {preset.id for preset in presets} == {TIME_ATTACK_PRESET_ID, GP_PRESET_ID}
    assert {(suite.preset_id, suite.preset_version, suite.status) for suite in suites} == {
        (TIME_ATTACK_PRESET_ID, 1, "not_created"),
        (GP_PRESET_ID, 1, "not_created"),
    }


def test_manager_store_creates_and_deletes_unused_custom_evaluation_preset(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")

    preset = store.create_evaluation_preset(
        name="Custom Time Attack",
        seed=7,
        renderer="angrylion",
        target=EvaluationTargetSpec(
            mode="time_attack_course",
            course_ids=("mute_city",),
            repeats_per_target=2,
        ),
    )

    assert preset.builtin is False
    assert preset.version == 1
    assert preset.target.vehicle_ids == ()
    assert store.delete_evaluation_preset(preset.id) is True
    assert store.get_evaluation_preset(preset.id) is None


def test_manager_store_rejects_deleting_used_evaluation_preset(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_latest_checkpoint(store, tmp_path)
    preset = store.create_evaluation_preset(
        name="Custom GP",
        seed=7,
        renderer="gliden64",
        target=EvaluationTargetSpec(
            mode="gp_course",
            course_ids=("mute_city",),
            difficulties=("master",),
            repeats_per_target=1,
        ),
    )
    store.create_evaluation(
        name="Eval 1",
        source_run_id=run.id,
        source_artifact="latest",
        policy_mode="deterministic",
        preset_id=preset.id,
        evaluations_root=tmp_path / "evaluations",
    )

    with pytest.raises(ValueError, match="referenced"):
        store.delete_evaluation_preset(preset.id)
