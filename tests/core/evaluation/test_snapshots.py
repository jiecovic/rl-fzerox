# tests/core/evaluation/test_snapshots.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.evaluation import (
    EvaluationCheckpointSource,
    snapshot_evaluation_checkpoint,
)
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import policy_artifact_metadata_path


def test_snapshot_evaluation_checkpoint_copies_policy_model_and_sidecars(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source"
    destination_dir = tmp_path / "eval" / "snapshot"
    checkpoint_dir = source_run_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    policy_path = checkpoint_dir / "policy.zip"
    policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(policy_path).write_text(
        '{"num_timesteps": 1234, "lineage_num_timesteps": 5678}\n',
        encoding="utf-8",
    )
    (checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename).write_bytes(b"engine-state")
    (checkpoint_dir / "future_sidecar.bin").write_bytes(b"future")

    snapshot = snapshot_evaluation_checkpoint(
        EvaluationCheckpointSource(
            run_id="run-a",
            run_name="Run A",
            run_dir=source_run_dir,
            artifact="latest",
        ),
        destination_dir=destination_dir,
    )

    destination_checkpoint_dir = destination_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    assert snapshot.source_run_id == "run-a"
    assert snapshot.local_num_timesteps == 1_234
    assert snapshot.lineage_num_timesteps == 5_678
    assert snapshot.source_policy_path == str(policy_path.resolve())
    assert snapshot.copied_policy_path == str(destination_checkpoint_dir / "policy.zip")
    assert snapshot.copied_model_path == str(destination_checkpoint_dir / "model.zip")
    assert sorted(path.name for path in destination_checkpoint_dir.iterdir()) == [
        "engine_tuning_state.json",
        "future_sidecar.bin",
        "model.zip",
        "policy.metadata.json",
        "policy.zip",
    ]
    assert (destination_checkpoint_dir / "policy.zip").read_bytes() == b"policy"


def test_snapshot_evaluation_checkpoint_copies_explicit_engine_tuning_sidecars(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "source"
    sidecar_dir = tmp_path / "sidecars"
    destination_dir = tmp_path / "eval" / "snapshot"
    checkpoint_dir = source_run_dir / RUN_LAYOUT.checkpoints_dirname / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    policy_path = checkpoint_dir / "policy.zip"
    policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(policy_path).write_text(
        '{"num_timesteps": 1234, "lineage_num_timesteps": 5678}\n',
        encoding="utf-8",
    )
    sidecar_dir.mkdir()
    engine_state_path = sidecar_dir / RUN_LAYOUT.engine_tuning_state_filename
    engine_model_path = sidecar_dir / RUN_LAYOUT.engine_tuning_model_filename
    engine_state_path.write_bytes(b"engine-state")
    engine_model_path.write_bytes(b"engine-model")

    snapshot_evaluation_checkpoint(
        EvaluationCheckpointSource(
            run_id="run-a",
            run_name="Run A",
            run_dir=source_run_dir,
            artifact="best",
            engine_tuning_state_path=engine_state_path,
            engine_tuning_model_path=engine_model_path,
        ),
        destination_dir=destination_dir,
    )

    destination_checkpoint_dir = destination_dir / RUN_LAYOUT.checkpoints_dirname / "best"
    assert (destination_checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename).read_bytes() == (
        b"engine-state"
    )
    assert (destination_checkpoint_dir / RUN_LAYOUT.engine_tuning_model_filename).read_bytes() == (
        b"engine-model"
    )


def test_snapshot_evaluation_checkpoint_rejects_missing_policy_metadata(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source"
    checkpoint_dir = source_run_dir / RUN_LAYOUT.checkpoints_dirname / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    (checkpoint_dir / "policy.zip").write_bytes(b"policy")

    with pytest.raises(ValueError, match="Could not determine checkpoint step"):
        snapshot_evaluation_checkpoint(
            EvaluationCheckpointSource(
                run_id="run-a",
                run_name="Run A",
                run_dir=source_run_dir,
                artifact="best",
            ),
            destination_dir=tmp_path / "eval",
        )


def test_snapshot_evaluation_checkpoint_rejects_non_empty_destination(
    tmp_path: Path,
) -> None:
    destination_dir = tmp_path / "eval"
    destination_dir.mkdir()
    (destination_dir / "existing.txt").write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        snapshot_evaluation_checkpoint(
            EvaluationCheckpointSource(
                run_id="run-a",
                run_name="Run A",
                run_dir=tmp_path / "source",
                artifact="latest",
            ),
            destination_dir=destination_dir,
        )
