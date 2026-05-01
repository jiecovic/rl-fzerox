# src/rl_fzerox/core/manager/manifest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.serialization import config_hash

MANIFEST_WARNING = (
    "SQLite is authoritative. Editing files in this directory does not change "
    "resume/fork behavior in the run manager."
)


@dataclass(frozen=True, slots=True)
class ManifestPaths:
    """Files written next to one DB-managed run."""

    manifest_json: Path
    manager_config_yaml: Path


def write_run_manifest(
    *,
    run_id: str,
    run_name: str,
    run_dir: Path,
    db_path: Path,
    config: ManagedRunConfig,
    created_at: str,
) -> ManifestPaths:
    """Write non-authoritative human-readable snapshots for one managed run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = ManifestPaths(
        manifest_json=run_dir / "manifest.json",
        manager_config_yaml=run_dir / "manager_config.yaml",
    )
    manifest_paths.manifest_json.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_name,
                "created_at": created_at,
                "config_hash": config_hash(config),
                "db_path": str(db_path),
                "warning": MANIFEST_WARNING,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    OmegaConf.save(
        config=OmegaConf.create(config.model_dump(mode="json")),
        f=str(manifest_paths.manager_config_yaml),
    )
    return manifest_paths
