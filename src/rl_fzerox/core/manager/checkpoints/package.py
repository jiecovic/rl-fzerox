# src/rl_fzerox/core/manager/checkpoints/package.py
"""Build maintainer checkpoint release bundles from local manager sources.

Packaging is intentionally separate from user-facing import/download flows. It
reads one trusted local run or evaluation snapshot from SQLite, copies only
curated checkpoint files into a release ZIP, and writes a manifest that future
importers can validate before touching local manager state.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.checkpoints.manifest import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleCheckpoint,
    CheckpointBundleCompatibility,
    CheckpointBundleFile,
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    CheckpointBundleSourceArtifact,
    serialize_checkpoint_bundle_manifest_json,
)
from rl_fzerox.core.manager.models import ManagedEvaluation, ManagedRun
from rl_fzerox.core.manager.projection.compat import fork_compatibility_signature
from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config
from rl_fzerox.core.manager.registry.common import slugify
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_json
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session import load_policy_artifact_metadata
from rl_fzerox.core.training.session.artifacts import (
    engine_tuning_checkpoint_path,
    engine_tuning_model_path,
    policy_artifact_metadata_path,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


@dataclass(frozen=True)
class CheckpointBundlePackageResult:
    """Result of writing one local checkpoint release bundle."""

    bundle_path: Path
    manifest: CheckpointBundleManifest


class CheckpointBundlePackageError(ValueError):
    """Raised when a local run cannot be packaged into a release checkpoint."""


@dataclass(frozen=True)
class _ArtifactRelativePaths:
    model: str
    policy: str


@dataclass(frozen=True)
class _PayloadFile:
    role: CheckpointBundleFileRole
    bundle_path: str
    content: bytes | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.content is None) == (self.source_path is None):
            raise CheckpointBundlePackageError(
                f"payload {self.bundle_path} must have exactly one content source"
            )

    def bytes(self) -> bytes:
        if self.content is not None:
            return self.content
        if self.source_path is None:
            raise CheckpointBundlePackageError(f"payload {self.bundle_path} has no source")
        return self.source_path.read_bytes()


_ARTIFACT_RELATIVE_PATHS: dict[CheckpointBundleSourceArtifact, _ArtifactRelativePaths] = {
    "latest": _ArtifactRelativePaths(
        model=RUN_LAYOUT.model_artifacts.latest,
        policy=RUN_LAYOUT.policy_artifacts.latest,
    ),
    "best": _ArtifactRelativePaths(
        model=RUN_LAYOUT.model_artifacts.best,
        policy=RUN_LAYOUT.policy_artifacts.best,
    ),
}


def default_checkpoint_bundle_path(
    *,
    checkpoint_id: str,
    version: str,
) -> Path:
    """Return the default ignored output path for a maintainer checkpoint bundle."""

    filename = f"rl-fzerox-checkpoint-{slugify(checkpoint_id) or 'checkpoint'}-{version}.zip"
    return project_root_dir() / "local" / "checkpoint_bundles" / filename


def package_checkpoint_bundle(
    *,
    store: ManagerStore,
    run_id: str,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    checkpoint_id: str | None = None,
    checkpoint_name: str | None = None,
    output_path: Path | None = None,
    allow_running: bool = False,
    overwrite: bool = False,
) -> CheckpointBundlePackageResult:
    """Package one exact run checkpoint into a distributable ZIP bundle."""

    run = store.get_run(run_id)
    if run is None:
        raise CheckpointBundlePackageError(f"run {run_id!r} does not exist")
    if run.status == "running" and not allow_running:
        raise CheckpointBundlePackageError("refusing to package a running run")
    if not run.run_dir.is_dir():
        raise CheckpointBundlePackageError(f"run directory does not exist: {run.run_dir}")

    normalized_version = _non_empty_text(version, field_name="version")
    resolved_checkpoint_id = _checkpoint_id(
        run=run,
        artifact=artifact,
        version=normalized_version,
        explicit_id=checkpoint_id,
    )
    resolved_output_path = (
        (
            output_path
            or default_checkpoint_bundle_path(
                checkpoint_id=resolved_checkpoint_id,
                version=normalized_version,
            )
        )
        .expanduser()
        .resolve()
    )
    if resolved_output_path.exists() and not overwrite:
        raise CheckpointBundlePackageError(f"output bundle already exists: {resolved_output_path}")

    payloads, train_config_sha256 = _payload_files(run=run, artifact=artifact)
    manifest = _manifest_for_run_payloads(
        store=store,
        run=run,
        artifact=artifact,
        version=normalized_version,
        checkpoint_id=resolved_checkpoint_id,
        checkpoint_name=checkpoint_name or run.name,
        payloads=payloads,
        train_config_sha256=train_config_sha256,
    )
    _write_bundle(resolved_output_path, manifest=manifest, payloads=payloads)
    return CheckpointBundlePackageResult(bundle_path=resolved_output_path, manifest=manifest)


def package_evaluation_checkpoint_bundle(
    *,
    store: ManagerStore,
    evaluation_id: str,
    version: str,
    checkpoint_id: str | None = None,
    checkpoint_name: str | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> CheckpointBundlePackageResult:
    """Package one completed evaluation snapshot into a distributable ZIP bundle."""

    evaluation = store.get_evaluation(evaluation_id)
    if evaluation is None:
        raise CheckpointBundlePackageError(f"evaluation {evaluation_id!r} does not exist")
    if evaluation.status == "running":
        raise CheckpointBundlePackageError("refusing to package a running evaluation")

    normalized_version = _non_empty_text(version, field_name="version")
    artifact = evaluation.checkpoint.artifact
    resolved_checkpoint_id = _evaluation_checkpoint_id(
        evaluation=evaluation,
        artifact=artifact,
        version=normalized_version,
        explicit_id=checkpoint_id,
    )
    resolved_output_path = (
        (
            output_path
            or default_checkpoint_bundle_path(
                checkpoint_id=resolved_checkpoint_id,
                version=normalized_version,
            )
        )
        .expanduser()
        .resolve()
    )
    if resolved_output_path.exists() and not overwrite:
        raise CheckpointBundlePackageError(f"output bundle already exists: {resolved_output_path}")

    policy_path = Path(evaluation.checkpoint.copied_policy_path).expanduser().resolve()
    compatibility_run_dir = evaluation.evaluation_dir / "checkpoint_snapshot"
    payloads, train_config_sha256 = _payload_files_for_checkpoint(
        config=evaluation.config,
        compatibility_run_id=evaluation.checkpoint.source_run_id or evaluation.id,
        compatibility_run_dir=compatibility_run_dir,
        policy_path=policy_path,
        model_path=_evaluation_model_path(evaluation),
        artifact_label=f"evaluation {evaluation.id} {artifact}",
        evaluation_metrics_path=_evaluation_metrics_path(evaluation),
    )
    manifest = _manifest_for_evaluation_payloads(
        store=store,
        evaluation=evaluation,
        artifact=artifact,
        version=normalized_version,
        checkpoint_id=resolved_checkpoint_id,
        checkpoint_name=checkpoint_name or evaluation.name,
        policy_path=policy_path,
        payloads=payloads,
        train_config_sha256=train_config_sha256,
    )
    _write_bundle(resolved_output_path, manifest=manifest, payloads=payloads)
    return CheckpointBundlePackageResult(bundle_path=resolved_output_path, manifest=manifest)


def _payload_files(
    *,
    run: ManagedRun,
    artifact: CheckpointBundleSourceArtifact,
) -> tuple[tuple[_PayloadFile, ...], str]:
    artifact_paths = _ARTIFACT_RELATIVE_PATHS[artifact]
    return _payload_files_for_checkpoint(
        config=run.config,
        compatibility_run_id=run.id,
        compatibility_run_dir=run.run_dir,
        policy_path=run.run_dir / artifact_paths.policy,
        model_path=run.run_dir / artifact_paths.model,
        artifact_label=artifact,
    )


def _payload_files_for_checkpoint(
    *,
    config: ManagedRunConfig,
    compatibility_run_id: str,
    compatibility_run_dir: Path,
    policy_path: Path,
    model_path: Path,
    artifact_label: str,
    evaluation_metrics_path: Path | None = None,
) -> tuple[tuple[_PayloadFile, ...], str]:
    metadata_path = policy_artifact_metadata_path(policy_path)
    _require_file(model_path, label=f"{artifact_label} model checkpoint")
    _require_file(policy_path, label=f"{artifact_label} policy checkpoint")
    _require_file(metadata_path, label=f"{artifact_label} policy metadata")

    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        raise CheckpointBundlePackageError(
            f"{artifact_label} policy metadata must include num_timesteps"
        )

    config_payload = config_json(config).encode("utf-8")
    payloads = [
        _PayloadFile("policy", "checkpoint/policy.zip", source_path=policy_path),
        _PayloadFile("model", "checkpoint/model.zip", source_path=model_path),
        _PayloadFile(
            "checkpoint_metadata",
            "checkpoint/policy.metadata.json",
            source_path=metadata_path,
        ),
        _PayloadFile("train_config", "config/train_config.json", content=config_payload),
    ]
    if evaluation_metrics_path is not None:
        _require_file(evaluation_metrics_path, label="evaluation metrics")
        payloads.append(
            _PayloadFile(
                "evaluation_metrics",
                "metrics/evaluation.json",
                source_path=evaluation_metrics_path,
            )
        )
    engine_state_path = engine_tuning_checkpoint_path(policy_path)
    if engine_state_path.is_file():
        payloads.append(
            _PayloadFile(
                "engine_tuning_state",
                "engine_tuning/state.json",
                source_path=engine_state_path,
            )
        )
    engine_model_path = engine_tuning_model_path(policy_path)
    if engine_model_path.is_file():
        payloads.append(
            _PayloadFile(
                "engine_tuning_model",
                "engine_tuning/model.pt",
                source_path=engine_model_path,
            )
        )

    return tuple(payloads), _sha256_bytes(config_payload)


def _manifest_for_run_payloads(
    *,
    store: ManagerStore,
    run: ManagedRun,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    checkpoint_id: str,
    checkpoint_name: str,
    payloads: tuple[_PayloadFile, ...],
    train_config_sha256: str,
) -> CheckpointBundleManifest:
    policy_path = run.run_dir / _ARTIFACT_RELATIVE_PATHS[artifact].policy
    return _manifest_for_checkpoint_payloads(
        store=store,
        config=run.config,
        compatibility_run_id=run.id,
        compatibility_run_dir=run.run_dir,
        policy_path=policy_path,
        artifact=artifact,
        version=version,
        checkpoint_id=checkpoint_id,
        checkpoint_name=checkpoint_name,
        source_run_id=run.id,
        source_run_name=run.name,
        created_at=None,
        payloads=payloads,
        train_config_sha256=train_config_sha256,
        lineage_step_offset=run.lineage_step_offset,
    )


def _manifest_for_evaluation_payloads(
    *,
    store: ManagerStore,
    evaluation: ManagedEvaluation,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    checkpoint_id: str,
    checkpoint_name: str,
    policy_path: Path,
    payloads: tuple[_PayloadFile, ...],
    train_config_sha256: str,
) -> CheckpointBundleManifest:
    return _manifest_for_checkpoint_payloads(
        store=store,
        config=evaluation.config,
        compatibility_run_id=evaluation.checkpoint.source_run_id or evaluation.id,
        compatibility_run_dir=evaluation.evaluation_dir / "checkpoint_snapshot",
        policy_path=policy_path,
        artifact=artifact,
        version=version,
        checkpoint_id=checkpoint_id,
        checkpoint_name=checkpoint_name,
        source_run_id=evaluation.checkpoint.source_run_id,
        source_run_name=evaluation.checkpoint.source_run_name,
        created_at=evaluation.finished_at or evaluation.created_at,
        payloads=payloads,
        train_config_sha256=train_config_sha256,
        lineage_step_offset=0,
    )


def _manifest_for_checkpoint_payloads(
    *,
    store: ManagerStore,
    config: ManagedRunConfig,
    compatibility_run_id: str,
    compatibility_run_dir: Path,
    policy_path: Path,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    checkpoint_id: str,
    checkpoint_name: str,
    source_run_id: str | None,
    source_run_name: str | None,
    created_at: str | None,
    payloads: tuple[_PayloadFile, ...],
    train_config_sha256: str,
    lineage_step_offset: int,
) -> CheckpointBundleManifest:
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        raise CheckpointBundlePackageError(f"{artifact} policy metadata is missing")
    lineage_steps = metadata.lineage_num_timesteps
    if lineage_steps is None and lineage_step_offset > 0:
        lineage_steps = lineage_step_offset + metadata.num_timesteps

    return CheckpointBundleManifest(
        format_name=CHECKPOINT_BUNDLE_LAYOUT.format_name,
        schema_version=CHECKPOINT_BUNDLE_LAYOUT.schema_version,
        exported_at=store.utc_now(),
        checkpoint=CheckpointBundleCheckpoint(
            id=checkpoint_id,
            name=_non_empty_text(checkpoint_name, field_name="checkpoint_name"),
            version=version,
            source_run_id=source_run_id,
            source_run_name=source_run_name,
            source_artifact=artifact,
            local_num_timesteps=metadata.num_timesteps,
            lineage_num_timesteps=lineage_steps,
            created_at=created_at,
        ),
        compatibility=_compatibility_for_config(
            config,
            run_id=compatibility_run_id,
            run_dir=compatibility_run_dir,
            train_config_sha256=train_config_sha256,
        ),
        files=tuple(_manifest_file(payload) for payload in payloads),
    )


def _compatibility_for_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    train_config_sha256: str,
) -> CheckpointBundleCompatibility:
    train_config = build_managed_train_app_config(config, run_id=run_id, run_dir=run_dir)
    signature = fork_compatibility_signature(train_config)
    return CheckpointBundleCompatibility(
        config_schema_version=int(config.version),
        training_algorithm=_string_value(signature.get("algorithm")),
        train_config_sha256=train_config_sha256,
        observation_space_sha256=_sha256_json(signature["observation"]),
        action_space_sha256=_sha256_json(signature["action"]),
        policy_signature_sha256=_sha256_json(signature["policy"]),
    )


def _manifest_file(payload: _PayloadFile) -> CheckpointBundleFile:
    data = payload.bytes()
    return CheckpointBundleFile(
        role=payload.role,
        path=payload.bundle_path,
        size_bytes=len(data),
        sha256=_sha256_bytes(data),
    )


def _write_bundle(
    bundle_path: Path,
    *,
    manifest: CheckpointBundleManifest,
    payloads: tuple[_PayloadFile, ...],
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = bundle_path.with_name(f".{bundle_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with zipfile.ZipFile(temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                CHECKPOINT_BUNDLE_LAYOUT.manifest_path,
                serialize_checkpoint_bundle_manifest_json(manifest),
            )
            for payload in payloads:
                archive.writestr(payload.bundle_path, payload.bytes())
        temporary_path.replace(bundle_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _checkpoint_id(
    *,
    run: ManagedRun,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    explicit_id: str | None,
) -> str:
    if explicit_id is not None:
        return _non_empty_text(explicit_id, field_name="checkpoint_id")
    return _non_empty_text(f"{run.id}-{artifact}-{version}", field_name="checkpoint_id")


def _evaluation_checkpoint_id(
    *,
    evaluation: ManagedEvaluation,
    artifact: CheckpointBundleSourceArtifact,
    version: str,
    explicit_id: str | None,
) -> str:
    if explicit_id is not None:
        return _non_empty_text(explicit_id, field_name="checkpoint_id")
    return _non_empty_text(
        f"{evaluation.id}-{artifact}-{version}",
        field_name="checkpoint_id",
    )


def _evaluation_model_path(evaluation: ManagedEvaluation) -> Path:
    if evaluation.checkpoint.copied_model_path is None:
        raise CheckpointBundlePackageError("evaluation checkpoint has no copied model path")
    return Path(evaluation.checkpoint.copied_model_path).expanduser().resolve()


def _evaluation_metrics_path(evaluation: ManagedEvaluation) -> Path | None:
    if evaluation.status != "completed" or evaluation.result_json_path is None:
        return None
    return evaluation.result_json_path.expanduser().resolve()


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise CheckpointBundlePackageError(f"missing {label}: {path}")
    if path.is_symlink():
        raise CheckpointBundlePackageError(f"{label} must not be a symlink: {path}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: object) -> str:
    payload = json.dumps(data, indent=None, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _sha256_bytes(payload)


def _non_empty_text(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise CheckpointBundlePackageError(f"{field_name} cannot be empty")
    return stripped


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None
