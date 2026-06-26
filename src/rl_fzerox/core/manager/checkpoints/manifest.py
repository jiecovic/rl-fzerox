# src/rl_fzerox/core/manager/checkpoints/manifest.py
"""Manifest schema for downloadable checkpoint bundles.

The manifest is the trust boundary for public checkpoint releases. It describes
the files inside a bundle before any importer writes local files or SQLite rows.
Validation stays intentionally strict so release bundles cannot smuggle ROMs,
emulator cores, baseline states, generated run directories, or unsafe paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    ValidationError,
    field_validator,
    model_validator,
)


@dataclass(frozen=True)
class CheckpointBundleLayout:
    format_name: str = "rl-fzerox-checkpoint-bundle"
    schema_version: int = 1
    manifest_path: str = "manifest.json"


CHECKPOINT_BUNDLE_LAYOUT = CheckpointBundleLayout()

CheckpointBundleFileRole = Literal[
    "policy",
    "model",
    "checkpoint_metadata",
    "train_config",
    "evaluation_metrics",
    "engine_tuning_state",
]

CheckpointBundleSourceArtifact = Literal["latest", "best", "final"]

_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PATH_PARTS = frozenset(
    {
        "baselines",
        "cores",
        "generated",
        "local",
        "roms",
        "runtime",
        "save_states",
        "savestates",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".dll",
        ".dylib",
        ".n64",
        ".rom",
        ".sav",
        ".so",
        ".srm",
        ".st",
        ".state",
        ".v64",
        ".z64",
    }
)
_ROLE_ROOTS: dict[CheckpointBundleFileRole, str] = {
    "policy": "checkpoint",
    "model": "checkpoint",
    "checkpoint_metadata": "checkpoint",
    "train_config": "config",
    "evaluation_metrics": "metrics",
    "engine_tuning_state": "engine_tuning",
}
_REQUIRED_ROLES = frozenset[CheckpointBundleFileRole]({"policy", "train_config"})
_SINGLETON_ROLES = frozenset[CheckpointBundleFileRole](
    {
        "policy",
        "model",
        "checkpoint_metadata",
        "train_config",
        "evaluation_metrics",
        "engine_tuning_state",
    }
)


class CheckpointBundleManifestError(ValueError):
    """Raised when raw manifest JSON cannot be parsed into a valid manifest."""


class CheckpointBundleModel(BaseModel):
    """Common strict model config for checkpoint bundle records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointBundleFile(CheckpointBundleModel):
    """One file carried by a checkpoint bundle."""

    role: CheckpointBundleFileRole
    path: str
    size_bytes: NonNegativeInt
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_bundle_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class CheckpointBundleCheckpoint(CheckpointBundleModel):
    """Human-facing identity and provenance for the published checkpoint."""

    id: str
    name: str
    version: str
    source_run_id: str | None = None
    source_run_name: str | None = None
    source_artifact: CheckpointBundleSourceArtifact = "best"
    local_num_timesteps: NonNegativeInt | None = None
    lineage_num_timesteps: NonNegativeInt | None = None
    created_at: str | None = None

    @field_validator("id", "name", "version", "source_run_id", "source_run_name", "created_at")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_text(value)


class CheckpointBundleCompatibility(CheckpointBundleModel):
    """Compatibility facts an importer can compare before installing a bundle."""

    app_version: str | None = None
    config_schema_version: NonNegativeInt | None = None
    training_algorithm: str | None = None
    policy_architecture: str | None = None
    train_config_sha256: str | None = None
    observation_space_sha256: str | None = None
    action_space_sha256: str | None = None

    @field_validator("app_version", "training_algorithm", "policy_architecture")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_text(value)

    @field_validator("train_config_sha256", "observation_space_sha256", "action_space_sha256")
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value)


class CheckpointBundleManifest(CheckpointBundleModel):
    """Top-level manifest stored as ``manifest.json`` in a checkpoint ZIP."""

    format_name: str = CHECKPOINT_BUNDLE_LAYOUT.format_name
    schema_version: int = CHECKPOINT_BUNDLE_LAYOUT.schema_version
    exported_at: str
    checkpoint: CheckpointBundleCheckpoint
    compatibility: CheckpointBundleCompatibility = CheckpointBundleCompatibility()
    files: tuple[CheckpointBundleFile, ...]

    @field_validator("exported_at")
    @classmethod
    def validate_exported_at(cls, value: str) -> str:
        return _validate_non_empty_text(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> CheckpointBundleManifest:
        if self.format_name != CHECKPOINT_BUNDLE_LAYOUT.format_name:
            raise ValueError(f"unsupported checkpoint bundle format {self.format_name!r}")
        if self.schema_version != CHECKPOINT_BUNDLE_LAYOUT.schema_version:
            raise ValueError(f"unsupported checkpoint bundle schema {self.schema_version}")
        _validate_files(self.files)
        return self


def parse_checkpoint_bundle_manifest_json(raw: str) -> CheckpointBundleManifest:
    """Parse raw manifest JSON and return a validated checkpoint bundle manifest."""

    try:
        return CheckpointBundleManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise CheckpointBundleManifestError(str(exc)) from exc


def serialize_checkpoint_bundle_manifest_json(manifest: CheckpointBundleManifest) -> str:
    """Serialize a manifest with deterministic JSON formatting for release tooling."""

    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _validate_files(files: tuple[CheckpointBundleFile, ...]) -> None:
    if not files:
        raise ValueError("checkpoint bundle must contain files")

    paths = [file.path for file in files]
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint bundle file paths must be unique")

    roles = [file.role for file in files]
    missing_roles = sorted(_REQUIRED_ROLES.difference(roles))
    if missing_roles:
        raise ValueError(f"checkpoint bundle is missing required file roles: {missing_roles}")

    for role in _SINGLETON_ROLES:
        if roles.count(role) > 1:
            raise ValueError(f"checkpoint bundle has more than one {role!r} file")

    for file in files:
        root = PurePosixPath(file.path).parts[0]
        expected_root = _ROLE_ROOTS[file.role]
        if root != expected_root:
            raise ValueError(
                f"checkpoint bundle file role {file.role!r} must live under {expected_root}/"
            )


def _validate_bundle_path(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("checkpoint bundle file path must be non-empty and trimmed")
    if "\\" in value:
        raise ValueError("checkpoint bundle file path must use POSIX separators")

    path = PurePosixPath(value)
    parts = path.parts
    if path.is_absolute() or ".." in parts or "." in parts or "" in parts:
        raise ValueError("checkpoint bundle file path must be a safe relative path")
    if len(parts) < 2:
        raise ValueError("checkpoint bundle file path must include a directory and filename")

    lower_parts = tuple(part.lower() for part in parts)
    if _FORBIDDEN_PATH_PARTS.intersection(lower_parts):
        raise ValueError("checkpoint bundle file path includes forbidden runtime asset storage")
    if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError("checkpoint bundle file path uses a forbidden runtime asset suffix")

    return value


def _validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not _HEX_SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("sha256 must be 64 hexadecimal characters")
    return normalized


def _validate_non_empty_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("text value must be non-empty")
    return stripped
