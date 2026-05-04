# src/rl_fzerox/core/manager/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.config import ManagedRunConfig

RunStatus = Literal["created", "running", "paused", "stopped", "finished", "failed"]
RunCommand = Literal["pause", "stop"]


@dataclass(frozen=True, slots=True)
class ManagedRunEvent:
    """One timestamped manager event for one run."""

    run_id: str
    created_at: str
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ManagedRunRuntime:
    """Latest sampled runtime state for one managed run."""

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


@dataclass(frozen=True, slots=True)
class ManagedRunMetricSample:
    """One historical metric sample captured for charts."""

    run_id: str
    created_at: str
    total_timesteps: int
    num_timesteps: int
    lineage_num_timesteps: int
    progress_fraction: float
    metrics: dict[str, float]
    fps: float | None = None
    episode_reward_mean: float | None = None
    episode_length_mean: float | None = None
    approx_kl: float | None = None
    entropy_loss: float | None = None
    value_loss: float | None = None
    policy_gradient_loss: float | None = None


@dataclass(frozen=True, slots=True)
class ManagedRun:
    """One immutable DB-managed run record."""

    id: str
    name: str
    status: RunStatus
    config: ManagedRunConfig
    config_hash: str
    run_dir: Path
    created_at: str
    lineage_id: str
    lineage_step_offset: int = 0
    parent_run_id: str | None = None
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_snapshot_dir: Path | None = None
    source_num_timesteps: int | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    runtime: ManagedRunRuntime | None = None
    pending_command: RunCommand | None = None


@dataclass(frozen=True, slots=True)
class ManagedRunTemplate:
    """One DB-backed editable starting point for future immutable runs."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ManagedRunDraft:
    """One SQLite-only editable run configuration draft."""

    id: str
    name: str
    config: ManagedRunConfig
    config_hash: str
    created_at: str
    updated_at: str
    source_run_id: str | None = None
    source_artifact: Literal["latest", "best"] | None = None
    source_snapshot_dir: Path | None = None
    source_num_timesteps: int | None = None
