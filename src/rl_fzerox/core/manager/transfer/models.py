# src/rl_fzerox/core/manager/transfer/models.py
"""Pydantic models for portable run bundle manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rl_fzerox.core.manager.models import RunStatus


class RunBundleFile(BaseModel):
    """One regular file stored under the exported run directory."""

    model_config = ConfigDict(frozen=True)

    path: str
    size_bytes: int


class RunBundleEvent(BaseModel):
    """One manager event copied into a portable run bundle."""

    model_config = ConfigDict(frozen=True)

    created_at: str
    kind: str
    message: str


class RunBundleRuntime(BaseModel):
    """Latest run-list runtime row copied into a portable run bundle."""

    model_config = ConfigDict(frozen=True)

    total_timesteps: int
    num_timesteps: int
    progress_fraction: float
    updated_at: str
    fps: float | None = None
    episode_reward_mean: float | None = None
    episode_length_mean: float | None = None
    approx_kl: float | None = None
    entropy_loss: float | None = None
    value_loss: float | None = None
    policy_gradient_loss: float | None = None


class RunBundleRecord(BaseModel):
    """Manager DB run row as serialized inside one bundle."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    status: RunStatus
    config: dict[str, object]
    run_dir: str
    lineage_id: str
    lineage_groups: tuple[str, ...] = ()
    lineage_step_offset: int = 0
    parent_run_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_snapshot_dir: str | None = None
    source_num_timesteps: int | None = None
    created_at: str
    started_at: str | None = None
    stopped_at: str | None = None
    runtime: RunBundleRuntime | None = None
    events: tuple[RunBundleEvent, ...] = ()


class RunBundleLayout(BaseModel):
    """Wire layout for run export archives."""

    model_config = ConfigDict(frozen=True)

    format_name: str = "rl-fzerox-run-bundle"
    schema_version: int = 1
    manifest_path: str = "run_export.json"
    payload_dir: str = "run"


class RunBundleManifest(BaseModel):
    """Portable manifest written next to the exported run payload."""

    model_config = ConfigDict(frozen=True)

    format_name: str
    schema_version: int
    exported_at: str
    project_root: str
    run: RunBundleRecord
    files: tuple[RunBundleFile, ...]


class RunBundleImportResult(BaseModel):
    """Result of importing a portable run bundle."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_dir: str
    imported_status: RunStatus
