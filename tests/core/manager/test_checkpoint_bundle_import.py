# tests/core/manager/test_checkpoint_bundle_import.py
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleImportError,
    CheckpointBundleManifest,
    import_checkpoint_bundle,
    parse_checkpoint_bundle_manifest_json,
)


def test_import_checkpoint_bundle_validates_and_extracts_payload(tmp_path: Path) -> None:
    bundle_path = tmp_path / "checkpoint.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)

    result = import_checkpoint_bundle(
        bundle_path=bundle_path,
        target_root=tmp_path / "manager" / "checkpoints",
    )

    assert result.checkpoint_id == "Blue Falcon Fine Tuned"
    assert result.version == "v1"
    assert (
        result.import_dir
        == tmp_path / "manager" / "checkpoints" / ("blue-falcon-fine-tuned") / "v1"
    )
    assert {path.relative_to(result.import_dir).as_posix() for path in result.files} == set(
        payloads
    )
    assert (result.import_dir / "checkpoint" / "policy.zip").read_bytes() == b"policy"
    installed_manifest = parse_checkpoint_bundle_manifest_json(
        (result.import_dir / CHECKPOINT_BUNDLE_LAYOUT.manifest_path).read_text(encoding="utf-8")
    )
    assert installed_manifest == manifest


def test_import_checkpoint_bundle_rejects_missing_payload(tmp_path: Path) -> None:
    bundle_path = tmp_path / "checkpoint.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(
        bundle_path,
        manifest=manifest,
        payloads={
            path: payload for path, payload in payloads.items() if path != "checkpoint/policy.zip"
        },
    )

    with pytest.raises(CheckpointBundleImportError, match="missing payload files"):
        import_checkpoint_bundle(
            bundle_path=bundle_path,
            target_root=tmp_path / "manager" / "checkpoints",
        )

    assert not (tmp_path / "manager" / "checkpoints" / "blue-falcon-fine-tuned").exists()


def test_import_checkpoint_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    bundle_path = tmp_path / "checkpoint.zip"
    payloads = _payloads()
    manifest = _manifest(payloads | {"checkpoint/policy.zip": b"POLICY"})
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)

    with pytest.raises(CheckpointBundleImportError, match="sha256 mismatch"):
        import_checkpoint_bundle(
            bundle_path=bundle_path,
            target_root=tmp_path / "manager" / "checkpoints",
        )


def test_import_checkpoint_bundle_rejects_unexpected_archive_member(tmp_path: Path) -> None:
    bundle_path = tmp_path / "checkpoint.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(
        bundle_path,
        manifest=manifest,
        payloads=payloads,
        extra_payloads={"../evil.txt": b"bad"},
    )

    with pytest.raises(CheckpointBundleImportError, match="unsafe archive member"):
        import_checkpoint_bundle(
            bundle_path=bundle_path,
            target_root=tmp_path / "manager" / "checkpoints",
        )


def test_import_checkpoint_bundle_refuses_existing_target(tmp_path: Path) -> None:
    bundle_path = tmp_path / "checkpoint.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)
    target_root = tmp_path / "manager" / "checkpoints"

    import_checkpoint_bundle(bundle_path=bundle_path, target_root=target_root)

    with pytest.raises(CheckpointBundleImportError, match="import target already exists"):
        import_checkpoint_bundle(bundle_path=bundle_path, target_root=target_root)


def _payloads() -> dict[str, bytes]:
    return {
        "checkpoint/policy.zip": b"policy",
        "checkpoint/model.zip": b"model",
        "checkpoint/policy.metadata.json": b'{"num_timesteps": 123}\n',
        "config/train_config.json": b'{"version": 1}\n',
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
    extra_payloads: dict[str, bytes] | None = None,
) -> None:
    with zipfile.ZipFile(bundle_path, mode="w") as archive:
        archive.writestr(
            CHECKPOINT_BUNDLE_LAYOUT.manifest_path,
            json.dumps(manifest.model_dump(mode="json")),
        )
        for path, payload in payloads.items():
            archive.writestr(path, payload)
        for path, payload in (extra_payloads or {}).items():
            archive.writestr(path, payload)
