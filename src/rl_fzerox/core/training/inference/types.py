# src/rl_fzerox/core/training/inference/types.py
"""Typed containers shared by inference loading and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class LoadedPolicy:
    """Resolved policy-only artifact metadata for watch mode."""

    run_dir: Path
    policy_path: Path
    artifact: str
    reload_source: Literal["artifact", "path"] = "artifact"
    model_path: Path | None = None
    device: str = "cpu"
    algorithm: str | None = None
    num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None
