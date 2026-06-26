# src/rl_fzerox/core/manager/checkpoints/__init__.py
"""Published checkpoint bundle metadata and validation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.manager.checkpoints.catalog import (
        CHECKPOINT_CATALOG_LAYOUT,
        CheckpointCatalog,
        CheckpointCatalogBundle,
        CheckpointCatalogEntry,
        CheckpointCatalogError,
        default_checkpoint_catalog_path,
        parse_checkpoint_catalog_json,
        serialize_checkpoint_catalog_json,
    )
    from rl_fzerox.core.manager.checkpoints.import_bundle import (
        CheckpointBundleImportError,
        CheckpointBundleImportResult,
        default_imported_checkpoint_root,
        import_checkpoint_bundle,
        read_checkpoint_bundle_manifest,
        validate_checkpoint_bundle_archive,
    )
    from rl_fzerox.core.manager.checkpoints.manifest import (
        CHECKPOINT_BUNDLE_LAYOUT,
        CheckpointBundleCheckpoint,
        CheckpointBundleCompatibility,
        CheckpointBundleFile,
        CheckpointBundleFileRole,
        CheckpointBundleManifest,
        CheckpointBundleManifestError,
        CheckpointBundleSourceArtifact,
        parse_checkpoint_bundle_manifest_json,
        serialize_checkpoint_bundle_manifest_json,
    )
    from rl_fzerox.core.manager.checkpoints.package import (
        CheckpointBundlePackageError,
        CheckpointBundlePackageResult,
        default_checkpoint_bundle_path,
        package_checkpoint_bundle,
        package_evaluation_checkpoint_bundle,
    )
    from rl_fzerox.core.manager.checkpoints.release import (
        CheckpointCatalogWriteResult,
        CheckpointReleaseError,
        checkpoint_catalog_entry_for_bundle,
        github_release_upload_command,
        make_github_release_asset_url,
        sha256_file,
        write_checkpoint_catalog_entry,
    )

_EXPORT_MODULES = {
    "CHECKPOINT_CATALOG_LAYOUT": "rl_fzerox.core.manager.checkpoints.catalog",
    "CHECKPOINT_BUNDLE_LAYOUT": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointCatalog": "rl_fzerox.core.manager.checkpoints.catalog",
    "CheckpointCatalogBundle": "rl_fzerox.core.manager.checkpoints.catalog",
    "CheckpointCatalogEntry": "rl_fzerox.core.manager.checkpoints.catalog",
    "CheckpointCatalogError": "rl_fzerox.core.manager.checkpoints.catalog",
    "CheckpointCatalogWriteResult": "rl_fzerox.core.manager.checkpoints.release",
    "CheckpointBundleCheckpoint": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundleCompatibility": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundleFile": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundleFileRole": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundleImportError": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "CheckpointBundleImportResult": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "CheckpointBundleManifest": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundleManifestError": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointBundlePackageError": "rl_fzerox.core.manager.checkpoints.package",
    "CheckpointBundlePackageResult": "rl_fzerox.core.manager.checkpoints.package",
    "CheckpointBundleSourceArtifact": "rl_fzerox.core.manager.checkpoints.manifest",
    "CheckpointReleaseError": "rl_fzerox.core.manager.checkpoints.release",
    "checkpoint_catalog_entry_for_bundle": "rl_fzerox.core.manager.checkpoints.release",
    "default_checkpoint_catalog_path": "rl_fzerox.core.manager.checkpoints.catalog",
    "default_checkpoint_bundle_path": "rl_fzerox.core.manager.checkpoints.package",
    "default_imported_checkpoint_root": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "github_release_upload_command": "rl_fzerox.core.manager.checkpoints.release",
    "import_checkpoint_bundle": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "make_github_release_asset_url": "rl_fzerox.core.manager.checkpoints.release",
    "package_checkpoint_bundle": "rl_fzerox.core.manager.checkpoints.package",
    "package_evaluation_checkpoint_bundle": "rl_fzerox.core.manager.checkpoints.package",
    "parse_checkpoint_catalog_json": "rl_fzerox.core.manager.checkpoints.catalog",
    "parse_checkpoint_bundle_manifest_json": "rl_fzerox.core.manager.checkpoints.manifest",
    "read_checkpoint_bundle_manifest": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "serialize_checkpoint_catalog_json": "rl_fzerox.core.manager.checkpoints.catalog",
    "serialize_checkpoint_bundle_manifest_json": "rl_fzerox.core.manager.checkpoints.manifest",
    "sha256_file": "rl_fzerox.core.manager.checkpoints.release",
    "validate_checkpoint_bundle_archive": "rl_fzerox.core.manager.checkpoints.import_bundle",
    "write_checkpoint_catalog_entry": "rl_fzerox.core.manager.checkpoints.release",
}

__all__ = [
    "CHECKPOINT_CATALOG_LAYOUT",
    "CHECKPOINT_BUNDLE_LAYOUT",
    "CheckpointCatalog",
    "CheckpointCatalogBundle",
    "CheckpointCatalogEntry",
    "CheckpointCatalogError",
    "CheckpointCatalogWriteResult",
    "CheckpointBundleCheckpoint",
    "CheckpointBundleCompatibility",
    "CheckpointBundleFile",
    "CheckpointBundleFileRole",
    "CheckpointBundleImportError",
    "CheckpointBundleImportResult",
    "CheckpointBundleManifest",
    "CheckpointBundleManifestError",
    "CheckpointBundlePackageError",
    "CheckpointBundlePackageResult",
    "CheckpointBundleSourceArtifact",
    "CheckpointReleaseError",
    "checkpoint_catalog_entry_for_bundle",
    "default_checkpoint_catalog_path",
    "default_checkpoint_bundle_path",
    "default_imported_checkpoint_root",
    "github_release_upload_command",
    "import_checkpoint_bundle",
    "make_github_release_asset_url",
    "package_checkpoint_bundle",
    "package_evaluation_checkpoint_bundle",
    "parse_checkpoint_catalog_json",
    "parse_checkpoint_bundle_manifest_json",
    "read_checkpoint_bundle_manifest",
    "serialize_checkpoint_catalog_json",
    "serialize_checkpoint_bundle_manifest_json",
    "sha256_file",
    "validate_checkpoint_bundle_archive",
    "write_checkpoint_catalog_entry",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
