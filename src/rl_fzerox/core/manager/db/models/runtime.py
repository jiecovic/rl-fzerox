# src/rl_fzerox/core/manager/db/models/runtime.py
"""ORM models for mutable run runtime state."""

from __future__ import annotations

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class RunRuntimeModel(ManagerBase):
    """Latest training metrics for one managed run."""

    __tablename__ = "run_runtime"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    total_timesteps: Mapped[int]
    num_timesteps: Mapped[int]
    progress_fraction: Mapped[float]
    updated_at: Mapped[str]
    fps: Mapped[float | None]
    episode_reward_mean: Mapped[float | None]
    episode_length_mean: Mapped[float | None]
    approx_kl: Mapped[float | None]
    entropy_loss: Mapped[float | None]
    value_loss: Mapped[float | None]
    policy_gradient_loss: Mapped[float | None]


class RunCommandModel(ManagerBase):
    """Pending command requested by the manager for one run worker."""

    __tablename__ = "run_commands"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    command: Mapped[str]
    requested_at: Mapped[str]


class RunWorkerModel(ManagerBase):
    """Current worker lease for one running managed run."""

    __tablename__ = "run_workers"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    launch_token: Mapped[str]
    pid: Mapped[int]
    launched_at: Mapped[str]
    heartbeat_at: Mapped[str]
