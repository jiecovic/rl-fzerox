# src/rl_fzerox/core/manager/checkpoints/catalog.py
"""Catalog schema for official downloadable checkpoint bundle entries.

The catalog is intentionally thinner than a bundle manifest. It tells the run
manager where to download a release ZIP and how to verify the ZIP as a whole;
the embedded bundle manifest remains the authority for the files inside that
ZIP after download.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    ValidationError,
    field_validator,
    model_validator,
)

from rl_fzerox.core.manager.checkpoints.manifest import CheckpointBundleManifest
from rl_fzerox.core.runtime_spec.paths import project_root_dir


@dataclass(frozen=True)
class CheckpointCatalogLayout:
    format_name: str = "rl-fzerox-checkpoint-catalog"
    schema_version: int = 1
    default_filename: str = "published_checkpoints.json"


CHECKPOINT_CATALOG_LAYOUT = CheckpointCatalogLayout()

_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointCatalogError(ValueError):
    """Raised when raw catalog JSON cannot be parsed into a valid catalog."""


class CheckpointCatalogModel(BaseModel):
    """Common strict model config for checkpoint catalog records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointCatalogBundle(CheckpointCatalogModel):
    """Download location and whole-bundle verification facts."""

    url: str
    filename: str
    size_bytes: NonNegativeInt
    sha256: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = _validate_non_empty_text(value)
        if not value.startswith("https://"):
            raise ValueError("checkpoint catalog bundle url must use https")
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = _validate_non_empty_text(value)
        if "/" in value or "\\" in value or value in {".", ".."} or not value.endswith(".zip"):
            raise ValueError("checkpoint catalog bundle filename must be a zip filename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class CheckpointCatalogEntry(CheckpointCatalogModel):
    """One official checkpoint release listed for download."""

    id: str
    version: str
    bundle: CheckpointCatalogBundle
    manifest: CheckpointBundleManifest

    @field_validator("id", "version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value)

    @model_validator(mode="after")
    def validate_entry(self) -> CheckpointCatalogEntry:
        checkpoint = self.manifest.checkpoint
        if self.id != checkpoint.id:
            raise ValueError("checkpoint catalog entry id must match bundle manifest")
        if self.version != checkpoint.version:
            raise ValueError("checkpoint catalog entry version must match bundle manifest")
        return self


class CheckpointCatalog(CheckpointCatalogModel):
    """Top-level official checkpoint catalog."""

    format_name: str = CHECKPOINT_CATALOG_LAYOUT.format_name
    schema_version: int = CHECKPOINT_CATALOG_LAYOUT.schema_version
    updated_at: str
    entries: tuple[CheckpointCatalogEntry, ...]

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        return _validate_non_empty_text(value)

    @model_validator(mode="after")
    def validate_catalog(self) -> CheckpointCatalog:
        if self.format_name != CHECKPOINT_CATALOG_LAYOUT.format_name:
            raise ValueError(f"unsupported checkpoint catalog format {self.format_name!r}")
        if self.schema_version != CHECKPOINT_CATALOG_LAYOUT.schema_version:
            raise ValueError(f"unsupported checkpoint catalog schema {self.schema_version}")
        if not self.entries:
            raise ValueError("checkpoint catalog must contain at least one entry")
        keys = [(entry.id, entry.version) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("checkpoint catalog entries must be unique by id and version")
        urls = [entry.bundle.url for entry in self.entries]
        if len(urls) != len(set(urls)):
            raise ValueError("checkpoint catalog bundle urls must be unique")
        return self


def default_checkpoint_catalog_path() -> Path:
    """Return the repo-tracked official checkpoint catalog path."""

    return project_root_dir() / CHECKPOINT_CATALOG_LAYOUT.default_filename


def parse_checkpoint_catalog_json(raw: str) -> CheckpointCatalog:
    """Parse raw catalog JSON and return a validated checkpoint catalog."""

    try:
        return CheckpointCatalog.model_validate_json(raw)
    except ValidationError as exc:
        raise CheckpointCatalogError(str(exc)) from exc


def serialize_checkpoint_catalog_json(catalog: CheckpointCatalog) -> str:
    """Serialize a catalog with deterministic JSON formatting for review."""

    return json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _validate_non_empty_text(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("checkpoint catalog text fields must be non-empty and trimmed")
    return value


def _validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not _HEX_SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("checkpoint catalog sha256 must be lowercase hex")
    return normalized
