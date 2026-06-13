# tests/core/manager/test_fork_source_artifacts.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.artifacts.fork_source import (
    is_complete_fork_source,
    snapshot_fork_source,
)
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import policy_artifact_metadata_path


def test_snapshot_fork_source_preserves_checkpoint_sidecars(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source"
    destination_dir = tmp_path / "snapshot"
    checkpoint_dir = source_run_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    checkpoint_dir.mkdir(parents=True)
    (source_run_dir / RUN_LAYOUT.config_filename).write_text("train: {}\n", encoding="utf-8")
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    policy_path = checkpoint_dir / "policy.zip"
    policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(policy_path).write_text(
        '{"num_timesteps": 816040}\n',
        encoding="utf-8",
    )
    (checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename).write_bytes(b"engine-state")
    (checkpoint_dir / RUN_LAYOUT.engine_tuning_model_filename).write_bytes(b"engine-model")
    (checkpoint_dir / "future_sidecar.bin").write_bytes(b"future")

    source_num_timesteps = snapshot_fork_source(
        source_run_dir=source_run_dir,
        artifact="latest",
        destination_dir=destination_dir,
    )

    destination_checkpoint_dir = destination_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    assert source_num_timesteps == 816_040
    assert sorted(path.name for path in destination_checkpoint_dir.iterdir()) == [
        "engine_tuning_model.pt",
        "engine_tuning_state.json",
        "future_sidecar.bin",
        "model.zip",
        "policy.metadata.json",
        "policy.zip",
    ]
    assert (destination_checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename).read_bytes() == (
        b"engine-state"
    )
    assert is_complete_fork_source(source_dir=destination_dir, artifact="latest")


def test_is_complete_fork_source_rejects_missing_policy_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "snapshot"
    checkpoint_dir = source_dir / RUN_LAYOUT.checkpoints_dirname / "latest"
    checkpoint_dir.mkdir(parents=True)
    (source_dir / RUN_LAYOUT.config_filename).write_text("train: {}\n", encoding="utf-8")
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    (checkpoint_dir / "policy.zip").write_bytes(b"policy")

    assert not is_complete_fork_source(source_dir=source_dir, artifact="latest")
