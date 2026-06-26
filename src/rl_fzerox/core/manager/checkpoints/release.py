# src/rl_fzerox/core/manager/checkpoints/release.py
"""Prepare official checkpoint release catalog entries.

This module owns the maintainer-side glue between a validated checkpoint ZIP and
the repo-tracked download catalog. It deliberately does not upload anything by
itself; scripts can use the returned GitHub CLI command when a maintainer wants
to publish the bundle asset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from rl_fzerox.core.manager.checkpoints.catalog import (
    CheckpointCatalog,
    CheckpointCatalogBundle,
    CheckpointCatalogEntry,
    parse_checkpoint_catalog_json,
    serialize_checkpoint_catalog_json,
)
from rl_fzerox.core.manager.checkpoints.import_bundle import validate_checkpoint_bundle_archive


class CheckpointReleaseError(ValueError):
    """Raised when release metadata cannot be prepared safely."""


@dataclass(frozen=True)
class CheckpointCatalogWriteResult:
    """Result of writing one release entry into a catalog file."""

    catalog: CheckpointCatalog
    entry: CheckpointCatalogEntry
    catalog_path: Path
    replaced: bool


def checkpoint_catalog_entry_for_bundle(
    *,
    bundle_path: Path,
    url: str,
) -> CheckpointCatalogEntry:
    """Build one catalog entry from a validated checkpoint release ZIP."""

    resolved_bundle_path = bundle_path.expanduser().resolve()
    if not resolved_bundle_path.is_file():
        raise CheckpointReleaseError(f"checkpoint bundle does not exist: {resolved_bundle_path}")
    manifest = validate_checkpoint_bundle_archive(bundle_path=resolved_bundle_path)
    return CheckpointCatalogEntry(
        id=manifest.checkpoint.id,
        version=manifest.checkpoint.version,
        bundle=CheckpointCatalogBundle(
            url=url,
            filename=resolved_bundle_path.name,
            size_bytes=resolved_bundle_path.stat().st_size,
            sha256=sha256_file(resolved_bundle_path),
        ),
        manifest=manifest,
    )


def write_checkpoint_catalog_entry(
    *,
    catalog_path: Path,
    entry: CheckpointCatalogEntry,
    updated_at: str | None = None,
) -> CheckpointCatalogWriteResult:
    """Insert or replace one checkpoint release entry in a catalog JSON file."""

    resolved_catalog_path = catalog_path.expanduser().resolve()
    entries = list(_catalog_entries(resolved_catalog_path))
    replaced = False
    for index, existing in enumerate(entries):
        if (existing.id, existing.version) == (entry.id, entry.version):
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    catalog = CheckpointCatalog(
        updated_at=updated_at or _utc_timestamp(),
        entries=tuple(entries),
    )
    resolved_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_catalog_path.write_text(serialize_checkpoint_catalog_json(catalog), encoding="utf-8")
    return CheckpointCatalogWriteResult(
        catalog=catalog,
        entry=entry,
        catalog_path=resolved_catalog_path,
        replaced=replaced,
    )


def make_github_release_asset_url(*, repo: str, release_tag: str, filename: str) -> str:
    """Return the public GitHub Release asset URL for one bundle file."""

    owner, repo_name = _github_repo_parts(repo)
    release_tag = _non_empty_text(release_tag, field_name="release tag")
    filename = _safe_filename(filename)
    return (
        f"https://github.com/{owner}/{repo_name}/releases/download/"
        f"{quote(release_tag, safe='')}/{quote(filename, safe='')}"
    )


def github_release_upload_command(
    *,
    repo: str,
    release_tag: str,
    bundle_path: Path,
    clobber: bool = False,
) -> tuple[str, ...]:
    """Return the GitHub CLI command for uploading one release bundle asset."""

    _github_repo_parts(repo)
    release_tag = _non_empty_text(release_tag, field_name="release tag")
    command = (
        "gh",
        "release",
        "upload",
        release_tag,
        str(bundle_path.expanduser().resolve()),
        "--repo",
        repo,
    )
    if clobber:
        return (*command, "--clobber")
    return command


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_entries(catalog_path: Path) -> tuple[CheckpointCatalogEntry, ...]:
    if not catalog_path.exists():
        return ()
    return parse_checkpoint_catalog_json(catalog_path.read_text(encoding="utf-8")).entries


def _github_repo_parts(repo: str) -> tuple[str, str]:
    repo = _non_empty_text(repo, field_name="repo")
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise CheckpointReleaseError("repo must use owner/name format")
    owner, repo_name = parts
    if any(char.isspace() for char in repo) or owner in {".", ".."} or repo_name in {".", ".."}:
        raise CheckpointReleaseError("repo must use owner/name format")
    return owner, repo_name


def _safe_filename(filename: str) -> str:
    filename = _non_empty_text(filename, field_name="filename")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise CheckpointReleaseError("filename must be a single file name")
    return filename


def _non_empty_text(value: str, *, field_name: str) -> str:
    if not value or value.strip() != value:
        raise CheckpointReleaseError(f"{field_name} must be non-empty and trimmed")
    return value


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
