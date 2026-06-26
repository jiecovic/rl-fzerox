# tests/core/manager/test_manager_store_checkpoints.py
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import select

from rl_fzerox.core.manager import ManagerStore, default_managed_run_config
from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
)
from rl_fzerox.core.manager.db.models.configs import ConfigSnapshotModel
from rl_fzerox.core.manager.storage.serialization import config_json
from rl_fzerox.core.training.runs import RUN_LAYOUT


def test_manager_store_imports_published_checkpoint_bundle(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    bundle_path = tmp_path / "blue-falcon.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)

    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)

    assert checkpoint.id == "blue-falcon-fine-tuned-v1"
    assert checkpoint.checkpoint_id == "Blue Falcon Fine Tuned"
    assert checkpoint.version == "v1"
    assert checkpoint.name == "72 x 96 IMPALA - like Blue Falcon Fine Tuned"
    assert checkpoint.config == default_managed_run_config()
    assert (
        checkpoint.import_dir
        == tmp_path / "manager" / "checkpoints" / ("blue-falcon-fine-tuned") / "v1"
    )
    assert checkpoint.run_id == "checkpoint-blue-falcon-fine-tuned-v1"
    assert checkpoint.policy_path == checkpoint.import_dir / RUN_LAYOUT.policy_artifacts.best
    assert checkpoint.model_path == checkpoint.import_dir / RUN_LAYOUT.model_artifacts.best
    assert not (checkpoint.import_dir / RUN_LAYOUT.policy_artifacts.latest).exists()
    assert not (checkpoint.import_dir / RUN_LAYOUT.model_artifacts.latest).exists()
    assert checkpoint.evaluation_metrics_path == checkpoint.import_dir / "metrics" / (
        "evaluation.json"
    )
    assert (
        checkpoint.engine_tuning_state_path
        == checkpoint.import_dir
        / Path(RUN_LAYOUT.policy_artifacts.best).parent
        / "engine_tuning_state.json"
    )
    assert checkpoint.source_bundle_path == bundle_path.resolve()
    assert checkpoint.source_bundle_sha256 == _sha256_file(bundle_path)
    assert checkpoint.source_run_id == "20260624-120927-f6fa7b32"
    assert checkpoint.source_run_name == "Blue Falcon Fine Tuned"
    assert checkpoint.source_artifact == "best"
    assert checkpoint.local_num_timesteps == 68_288_256
    assert checkpoint.lineage_num_timesteps == 1_979_774_040
    assert checkpoint.exported_at == manifest.exported_at
    assert checkpoint.imported_at == checkpoint.updated_at
    assert checkpoint.policy_path.read_bytes() == b"policy"

    assert store.get_published_checkpoint(checkpoint.id) == checkpoint
    assert store.list_published_checkpoints() == (checkpoint,)
    run = store.get_run(checkpoint.run_id)
    assert run is not None
    assert run.status == "archived"
    assert run.run_dir == checkpoint.import_dir
    assert run.lineage_id == checkpoint.id
    assert run.lineage_step_offset == 1_911_485_784
    assert run.source_num_timesteps == checkpoint.local_num_timesteps


def test_manager_store_published_checkpoint_has_import_config_snapshot(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    bundle_path = tmp_path / "blue-falcon.zip"
    _write_bundle(bundle_path, manifest=_manifest(_payloads()), payloads=_payloads())

    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)

    with store._orm_session() as session:
        snapshots = tuple(
            session.scalars(select(ConfigSnapshotModel).where(ConfigSnapshotModel.kind == "import"))
        )

    assert len(snapshots) == 1
    assert snapshots[0].kind == "import"
    assert snapshots[0].config_hash == checkpoint.config_hash
    assert snapshots[0].config_json == config_json(default_managed_run_config())


def test_manager_store_rejects_duplicate_published_checkpoint(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    bundle_path = tmp_path / "blue-falcon.zip"
    payloads = _payloads()
    _write_bundle(bundle_path, manifest=_manifest(payloads), payloads=payloads)

    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)

    with pytest.raises(ValueError, match="already exists"):
        store.import_published_checkpoint_bundle(
            bundle_path=bundle_path,
            target_root=tmp_path / "other-checkpoints",
        )

    assert checkpoint.import_dir.is_dir()
    assert not (tmp_path / "other-checkpoints").exists()


def test_manager_store_resolves_published_checkpoint_policy_source(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    bundle_path = tmp_path / "blue-falcon.zip"
    payloads = _payloads()
    _write_bundle(bundle_path, manifest=_manifest(payloads), payloads=payloads)
    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)

    policy_source = store.resolve_policy_source(
        policy_source_kind="checkpoint",
        policy_source_id=checkpoint.id,
        policy_artifact="best",
        require_policy_artifact=True,
    )

    assert policy_source.kind == "checkpoint"
    assert policy_source.id == checkpoint.id
    assert policy_source.mutable is False
    assert policy_source.source_dir == checkpoint.import_dir
    assert policy_source.policy_path == checkpoint.policy_path
    assert policy_source.model_path == checkpoint.model_path
    assert policy_source.engine_tuning_state_path == checkpoint.engine_tuning_state_path
    assert policy_source.engine_tuning_model_path == checkpoint.engine_tuning_model_path
    assert policy_source.lineage_num_timesteps == checkpoint.lineage_num_timesteps


def test_manager_store_save_course_setup_accepts_published_checkpoint(
    tmp_path: Path,
) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    bundle_path = tmp_path / "blue-falcon.zip"
    payloads = _payloads()
    _write_bundle(bundle_path, manifest=_manifest(payloads), payloads=payloads)
    checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)
    save_game = store.create_save_game(name="Unlock Save", save_games_root=tmp_path / "saves")

    setup = store.upsert_save_course_setup(
        save_game_id=save_game.id,
        cup_id="jack",
        course_id="mute_city",
        policy_source_kind="checkpoint",
        policy_source_id=checkpoint.id,
        policy_artifact="best",
    )

    assert setup.policy_source_kind == "checkpoint"
    assert setup.policy_source_id == checkpoint.id
    assert store.list_save_course_setups(save_game.id) == (setup,)


def _payloads() -> dict[str, bytes]:
    return {
        "checkpoint/policy.zip": b"policy",
        "checkpoint/model.zip": b"model",
        "checkpoint/policy.metadata.json": (
            b'{"num_timesteps": 68288256, "lineage_num_timesteps": 1979774040}\n'
        ),
        "config/train_config.json": config_json(default_managed_run_config()).encode("utf-8"),
        "engine_tuning/state.json": b"{}\n",
        "metrics/evaluation.json": b'{"finish_rate": 0.875}\n',
    }


def _manifest(payloads: dict[str, bytes]) -> CheckpointBundleManifest:
    return CheckpointBundleManifest(
        exported_at="2026-06-26T10:00:00+00:00",
        checkpoint=CheckpointBundleCheckpoint(
            id="Blue Falcon Fine Tuned",
            name="72 x 96 IMPALA - like Blue Falcon Fine Tuned",
            version="v1",
            source_run_id="20260624-120927-f6fa7b32",
            source_run_name="Blue Falcon Fine Tuned",
            source_artifact="best",
            local_num_timesteps=68_288_256,
            lineage_num_timesteps=1_979_774_040,
            created_at="2026-06-26T10:26:03+00:00",
        ),
        files=tuple(_file(path, payload) for path, payload in payloads.items()),
    )


def _file(path: str, payload: bytes) -> CheckpointBundleFile:
    role_by_path: dict[str, CheckpointBundleFileRole] = {
        "checkpoint/policy.zip": "policy",
        "checkpoint/model.zip": "model",
        "checkpoint/policy.metadata.json": "checkpoint_metadata",
        "config/train_config.json": "train_config",
        "engine_tuning/state.json": "engine_tuning_state",
        "metrics/evaluation.json": "evaluation_metrics",
    }
    return CheckpointBundleFile(
        role=role_by_path[path],
        path=path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_bundle(
    bundle_path: Path,
    *,
    manifest: CheckpointBundleManifest,
    payloads: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(bundle_path, mode="w") as archive:
        archive.writestr(
            CHECKPOINT_BUNDLE_LAYOUT.manifest_path,
            json.dumps(manifest.model_dump(mode="json")),
        )
        for path, payload in payloads.items():
            archive.writestr(path, payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
