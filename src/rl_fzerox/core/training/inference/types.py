# src/rl_fzerox/core/training/inference/types.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedPolicy:
    """Resolved policy-only artifact metadata for watch mode."""

    run_dir: Path
    policy_path: Path
    artifact: str
    device: str = "cpu"
    algorithm: str | None = None
    num_timesteps: int | None = None
    lineage_num_timesteps: int | None = None
