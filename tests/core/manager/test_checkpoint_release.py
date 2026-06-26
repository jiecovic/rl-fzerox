# tests/core/manager/test_checkpoint_release.py
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    checkpoint_catalog_entry_for_bundle,
    github_release_upload_command,
    make_github_release_asset_url,
    parse_checkpoint_catalog_json,
    write_checkpoint_catalog_entry,
)


def test_checkpoint_catalog_entry_for_bundle_uses_whole_zip_hash(tmp_path: Path) -> None:
    bundle_path = tmp_path / "blue-falcon-v1.zip"
    payloads = _payloads()
    manifest = _manifest(payloads)
    _write_bundle(bundle_path, manifest=manifest, payloads=payloads)

    entry = checkpoint_catalog_entry_for_bundle(
        bundle_path=bundle_path,
        url="https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints-v1/blue.zip",
    )

    assert entry.id == "blue-falcon-fine-tuned"
    assert entry.version == "v1"
    assert entry.bundle.filename == "blue-falcon-v1.zip"
    assert entry.bundle.size_bytes == bundle_path.stat().st_size
    assert entry.bundle.sha256 == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert entry.manifest == manifest


def test_write_checkpoint_catalog_entry_replaces_existing_version(tmp_path: Path) -> None:
    catalog_path = tmp_path / "published_checkpoints.json"
    old_bundle_path = tmp_path / "old.zip"
    new_bundle_path = tmp_path / "new.zip"
    old_payloads = _payloads(policy=b"old policy")
    new_payloads = _payloads(policy=b"new policy")
    _write_bundle(old_bundle_path, manifest=_manifest(old_payloads), payloads=old_payloads)
    _write_bundle(new_bundle_path, manifest=_manifest(new_payloads), payloads=new_payloads)
    old_entry = checkpoint_catalog_entry_for_bundle(
        bundle_path=old_bundle_path,
        url="https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints-v1/old.zip",
    )
    new_entry = checkpoint_catalog_entry_for_bundle(
        bundle_path=new_bundle_path,
        url="https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints-v1/new.zip",
    )

    write_checkpoint_catalog_entry(
        catalog_path=catalog_path,
        entry=old_entry,
        updated_at="2026-06-26T10:00:00+00:00",
    )
    result = write_checkpoint_catalog_entry(
        catalog_path=catalog_path,
        entry=new_entry,
        updated_at="2026-06-26T11:00:00+00:00",
    )

    parsed = parse_checkpoint_catalog_json(catalog_path.read_text(encoding="utf-8"))
    assert result.replaced is True
    assert parsed.updated_at == "2026-06-26T11:00:00+00:00"
    assert len(parsed.entries) == 1
    assert parsed.entries[0].bundle.sha256 == new_entry.bundle.sha256


def test_github_release_url_and_upload_command() -> None:
    url = make_github_release_asset_url(
        repo="jiecovic/rl-fzerox",
        release_tag="checkpoints v1",
        filename="blue falcon.zip",
    )
    command = github_release_upload_command(
        repo="jiecovic/rl-fzerox",
        release_tag="checkpoints-v1",
        bundle_path=Path("/tmp/blue.zip"),
        clobber=True,
    )

    assert url == (
        "https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints%20v1/blue%20falcon.zip"
    )
    assert command == (
        "gh",
        "release",
        "upload",
        "checkpoints-v1",
        "/tmp/blue.zip",
        "--repo",
        "jiecovic/rl-fzerox",
        "--clobber",
    )


def _payloads(*, policy: bytes = b"policy") -> dict[str, bytes]:
    return {
        "checkpoint/policy.zip": policy,
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
            id="blue-falcon-fine-tuned",
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
