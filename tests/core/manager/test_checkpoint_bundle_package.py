# tests/core/manager/test_checkpoint_bundle_package.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundlePackageError,
    package_checkpoint_bundle,
    parse_checkpoint_bundle_manifest_json,
)
from rl_fzerox.core.manager.storage.serialization import config_json
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.artifacts import policy_artifact_metadata_path


def test_package_checkpoint_bundle_writes_manifest_and_payloads(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_checkpoint(store, tmp_path, artifact="best")
    bundle_path = tmp_path / "bundle.zip"

    result = package_checkpoint_bundle(
        store=store,
        run_id=run.id,
        artifact="best",
        version="v1",
        checkpoint_id="golden-fox-v1",
        output_path=bundle_path,
    )

    assert result.bundle_path == bundle_path
    with zipfile.ZipFile(bundle_path) as archive:
        assert sorted(archive.namelist()) == [
            "checkpoint/model.zip",
            "checkpoint/policy.metadata.json",
            "checkpoint/policy.zip",
            "config/train_config.json",
            "engine_tuning/state.json",
            "manifest.json",
        ]
        manifest = parse_checkpoint_bundle_manifest_json(
            archive.read(CHECKPOINT_BUNDLE_LAYOUT.manifest_path).decode("utf-8")
        )
        assert archive.read("checkpoint/policy.zip") == b"policy"
        assert archive.read("checkpoint/model.zip") == b"model"
        assert json.loads(archive.read("config/train_config.json")) == json.loads(
            config_json(run.config)
        )

    assert manifest.checkpoint.id == "golden-fox-v1"
    assert manifest.checkpoint.source_run_id == run.id
    assert manifest.checkpoint.source_artifact == "best"
    assert manifest.checkpoint.local_num_timesteps == 123_456
    assert manifest.checkpoint.lineage_num_timesteps == 10_123_456
    assert manifest.compatibility.config_schema_version == 1
    assert manifest.compatibility.train_config_sha256 is not None
    assert {file.role for file in manifest.files} == {
        "policy",
        "model",
        "checkpoint_metadata",
        "train_config",
        "engine_tuning_state",
    }


def test_package_checkpoint_bundle_requires_exact_artifact(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_checkpoint(store, tmp_path, artifact="best")

    with pytest.raises(CheckpointBundlePackageError, match="missing latest model checkpoint"):
        package_checkpoint_bundle(
            store=store,
            run_id=run.id,
            artifact="latest",
            version="v1",
            output_path=tmp_path / "bundle.zip",
        )


def test_package_checkpoint_bundle_requires_policy_metadata(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_checkpoint(store, tmp_path, artifact="best")
    policy_path = run.run_dir / RUN_LAYOUT.policy_artifacts.best
    policy_artifact_metadata_path(policy_path).unlink()

    with pytest.raises(CheckpointBundlePackageError, match="missing best policy metadata"):
        package_checkpoint_bundle(
            store=store,
            run_id=run.id,
            artifact="best",
            version="v1",
            output_path=tmp_path / "bundle.zip",
        )


def test_package_checkpoint_bundle_refuses_running_run(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_checkpoint(store, tmp_path, artifact="best")
    store.update_run_status(run_id=run.id, status="running", message="running")

    with pytest.raises(CheckpointBundlePackageError, match="running run"):
        package_checkpoint_bundle(
            store=store,
            run_id=run.id,
            artifact="best",
            version="v1",
            output_path=tmp_path / "bundle.zip",
        )


def test_package_checkpoint_bundle_refuses_overwrite_by_default(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    run = _create_run_with_checkpoint(store, tmp_path, artifact="best")
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"existing")

    with pytest.raises(CheckpointBundlePackageError, match="already exists"):
        package_checkpoint_bundle(
            store=store,
            run_id=run.id,
            artifact="best",
            version="v1",
            output_path=bundle_path,
        )


def _create_run_with_checkpoint(
    store: ManagerStore,
    tmp_path: Path,
    *,
    artifact: str,
):
    run = store.create_run(
        run_id="run-a",
        name="Golden Fox Release Candidate",
        config=default_managed_run_config(),
        explicit_run_dir=tmp_path / "runs" / "lineage-a" / "run-a",
        lineage_id="lineage-a",
        lineage_step_offset=10_000_000,
    )
    checkpoint_dir = run.run_dir / RUN_LAYOUT.checkpoints_dirname / artifact
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.zip").write_bytes(b"model")
    policy_path = checkpoint_dir / "policy.zip"
    policy_path.write_bytes(b"policy")
    policy_artifact_metadata_path(policy_path).write_text(
        json.dumps(
            {
                "num_timesteps": 123_456,
                "lineage_num_timesteps": 10_123_456,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / RUN_LAYOUT.engine_tuning_state_filename).write_text(
        "{}\n",
        encoding="utf-8",
    )
    store.update_run_status(run_id=run.id, status="stopped", message="stopped")
    loaded = store.get_run(run.id)
    assert loaded is not None
    return loaded
