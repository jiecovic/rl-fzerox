# tests/core/manager/test_checkpoint_catalog.py
from __future__ import annotations

import pytest

from rl_fzerox.core.manager.checkpoints import (
    CHECKPOINT_CATALOG_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    CheckpointCatalog,
    CheckpointCatalogBundle,
    CheckpointCatalogEntry,
    CheckpointCatalogError,
    default_checkpoint_catalog_path,
    parse_checkpoint_catalog_json,
    serialize_checkpoint_catalog_json,
)

_HASH = "a" * 64


def test_checkpoint_catalog_accepts_minimal_entry() -> None:
    catalog = _catalog()

    assert catalog.format_name == CHECKPOINT_CATALOG_LAYOUT.format_name
    assert catalog.schema_version == CHECKPOINT_CATALOG_LAYOUT.schema_version
    assert catalog.entries[0].bundle.sha256 == _HASH


def test_checkpoint_catalog_json_roundtrip_normalizes_hash() -> None:
    catalog = _catalog(bundle_hash="B" * 64)

    raw = serialize_checkpoint_catalog_json(catalog)
    parsed = parse_checkpoint_catalog_json(raw)

    assert parsed == catalog
    assert parsed.entries[0].bundle.sha256 == "b" * 64


def test_checkpoint_catalog_rejects_manifest_id_mismatch() -> None:
    with pytest.raises(ValueError, match="entry id must match"):
        CheckpointCatalogEntry(
            id="other-checkpoint",
            version="v1",
            bundle=_bundle(),
            manifest=_manifest(),
        )


def test_checkpoint_catalog_rejects_duplicate_entries() -> None:
    entry = _entry()

    with pytest.raises(ValueError, match="unique by id and version"):
        CheckpointCatalog(updated_at="2026-06-26T12:00:00+00:00", entries=(entry, entry))


def test_checkpoint_catalog_rejects_non_https_url() -> None:
    with pytest.raises(ValueError, match="must use https"):
        CheckpointCatalogBundle(
            url="http://example.invalid/checkpoint.zip",
            filename="checkpoint.zip",
            size_bytes=1,
            sha256=_HASH,
        )


def test_checkpoint_catalog_parse_wraps_pydantic_errors() -> None:
    with pytest.raises(CheckpointCatalogError, match="json_invalid"):
        parse_checkpoint_catalog_json("{")


def test_official_checkpoint_catalog_file_parses() -> None:
    catalog = parse_checkpoint_catalog_json(default_checkpoint_catalog_path().read_text())

    assert len(catalog.entries) == 1
    entry = catalog.entries[0]
    assert entry.id == "blue-falcon-all-cups"
    assert entry.version == "v1"
    assert entry.bundle.filename == "rl-fzerox-checkpoint-blue-falcon-all-cups-v1.zip"
    assert entry.manifest.checkpoint.lineage_num_timesteps == 1_979_774_040


def _catalog(*, bundle_hash: str = _HASH) -> CheckpointCatalog:
    return CheckpointCatalog(
        updated_at="2026-06-26T12:00:00+00:00",
        entries=(_entry(bundle_hash=bundle_hash),),
    )


def _entry(*, bundle_hash: str = _HASH) -> CheckpointCatalogEntry:
    return CheckpointCatalogEntry(
        id="blue-falcon-fine-tuned",
        version="v1",
        bundle=_bundle(bundle_hash=bundle_hash),
        manifest=_manifest(),
    )


def _bundle(*, bundle_hash: str = _HASH) -> CheckpointCatalogBundle:
    return CheckpointCatalogBundle(
        url=(
            "https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints-v1/checkpoint.zip"
        ),
        filename="checkpoint.zip",
        size_bytes=1,
        sha256=bundle_hash,
    )


def _manifest() -> CheckpointBundleManifest:
    return CheckpointBundleManifest(
        exported_at="2026-06-26T12:00:00+00:00",
        checkpoint=CheckpointBundleCheckpoint(
            id="blue-falcon-fine-tuned",
            name="72 x 96 IMPALA - like Blue Falcon Fine Tuned",
            version="v1",
        ),
        files=(
            _file("policy", "checkpoint/policy.zip"),
            _file("model", "checkpoint/model.zip"),
            _file("checkpoint_metadata", "checkpoint/policy.metadata.json"),
            _file("train_config", "config/train_config.json"),
        ),
    )


def _file(role: CheckpointBundleFileRole, path: str) -> CheckpointBundleFile:
    return CheckpointBundleFile(role=role, path=path, size_bytes=1, sha256=_HASH)
