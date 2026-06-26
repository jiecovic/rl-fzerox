# tests/core/manager/test_checkpoint_bundle_manifest.py
from __future__ import annotations

import pytest

from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    CheckpointBundleManifestError,
    parse_checkpoint_bundle_manifest_json,
    serialize_checkpoint_bundle_manifest_json,
)

_HASH = "a" * 64


def test_checkpoint_bundle_manifest_accepts_minimal_release() -> None:
    manifest = _manifest()

    assert manifest.format_name == CHECKPOINT_BUNDLE_LAYOUT.format_name
    assert manifest.schema_version == CHECKPOINT_BUNDLE_LAYOUT.schema_version
    assert manifest.files[0].sha256 == _HASH


def test_checkpoint_bundle_manifest_json_roundtrip_normalizes_hash() -> None:
    manifest = _manifest(policy_hash="B" * 64)

    raw = serialize_checkpoint_bundle_manifest_json(manifest)
    parsed = parse_checkpoint_bundle_manifest_json(raw)

    assert parsed == manifest
    assert parsed.files[0].sha256 == "b" * 64


@pytest.mark.parametrize(
    "path",
    [
        "/checkpoint/policy.zip",
        "../checkpoint/policy.zip",
        "checkpoint/../policy.zip",
        "checkpoint\\policy.zip",
        "policy.zip",
        " checkpoint/policy.zip",
    ],
)
def test_checkpoint_bundle_manifest_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="checkpoint bundle file path"):
        CheckpointBundleFile(role="policy", path=path, size_bytes=1, sha256=_HASH)


@pytest.mark.parametrize(
    "path",
    [
        "checkpoint/fzerox.n64",
        "checkpoint/core.so",
        "checkpoint/save.state",
        "roms/fzerox.n64",
        "baselines/mute-city.state",
    ],
)
def test_checkpoint_bundle_manifest_rejects_runtime_assets(path: str) -> None:
    with pytest.raises(ValueError, match="forbidden runtime asset"):
        CheckpointBundleFile(role="policy", path=path, size_bytes=1, sha256=_HASH)


def test_checkpoint_bundle_manifest_rejects_bad_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        CheckpointBundleFile(role="policy", path="checkpoint/policy.zip", size_bytes=1, sha256="x")


def test_checkpoint_bundle_manifest_rejects_duplicate_paths() -> None:
    with pytest.raises(ValueError, match="paths must be unique"):
        _manifest(
            files=(
                _file("policy", "checkpoint/policy.zip"),
                _file("train_config", "checkpoint/policy.zip"),
            )
        )


def test_checkpoint_bundle_manifest_rejects_missing_policy() -> None:
    with pytest.raises(ValueError, match="missing required file roles"):
        _manifest(
            files=(
                _file("model", "checkpoint/model.zip"),
                _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
                _file("train_config", "config/train_config.json"),
            )
        )


def test_checkpoint_bundle_manifest_rejects_missing_train_config() -> None:
    with pytest.raises(ValueError, match="missing required file roles"):
        _manifest(
            files=(
                _file("policy", "checkpoint/policy.zip"),
                _file("model", "checkpoint/model.zip"),
                _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
            )
        )


def test_checkpoint_bundle_manifest_rejects_duplicate_singleton_role() -> None:
    with pytest.raises(ValueError, match="more than one 'policy' file"):
        _manifest(
            files=(
                _file("policy", "checkpoint/policy.zip"),
                _file("policy", "checkpoint/policy-copy.zip"),
                _file("model", "checkpoint/model.zip"),
                _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
                _file("train_config", "config/train_config.json"),
            )
        )


def test_checkpoint_bundle_manifest_rejects_role_in_wrong_directory() -> None:
    with pytest.raises(ValueError, match="must live under checkpoint"):
        _manifest(
            files=(
                _file("policy", "config/policy.zip"),
                _file("model", "checkpoint/model.zip"),
                _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
                _file("train_config", "config/train_config.json"),
            )
        )


def test_checkpoint_bundle_manifest_rejects_unsupported_wire_version() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint bundle schema"):
        _manifest(schema_version=2)


def test_checkpoint_bundle_manifest_parse_wraps_pydantic_errors() -> None:
    with pytest.raises(CheckpointBundleManifestError, match="json_invalid"):
        parse_checkpoint_bundle_manifest_json("{")


def _manifest(
    *,
    schema_version: int = CHECKPOINT_BUNDLE_LAYOUT.schema_version,
    policy_hash: str = _HASH,
    files: tuple[CheckpointBundleFile, ...] | None = None,
) -> CheckpointBundleManifest:
    return CheckpointBundleManifest(
        format_name=CHECKPOINT_BUNDLE_LAYOUT.format_name,
        schema_version=schema_version,
        exported_at="2026-06-26T10:00:00+00:00",
        checkpoint=CheckpointBundleCheckpoint(
            id="impala-2b-golden-fox-v1",
            name="IMPALA 2B Golden Fox",
            version="v1",
            source_run_id="20260625-130715-f6391ad6",
            source_run_name="72 x 96 IMPALA-like Golden Fox Test",
            local_num_timesteps=500_000_000,
            lineage_num_timesteps=1_976_925_432,
        ),
        files=files
        or (
            _file("policy", "checkpoint/policy.zip", sha256=policy_hash),
            _file("model", "checkpoint/model.zip"),
            _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
            _file("train_config", "config/train_config.json"),
        ),
    )


def _file(
    role: CheckpointBundleFileRole,
    path: str,
    *,
    sha256: str = _HASH,
) -> CheckpointBundleFile:
    return CheckpointBundleFile(role=role, path=path, size_bytes=1, sha256=sha256)
