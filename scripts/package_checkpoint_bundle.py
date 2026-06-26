# scripts/package_checkpoint_bundle.py
"""Maintainer tool for building downloadable checkpoint release bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_fzerox.core.manager.checkpoints import (  # noqa: E402
    CheckpointBundlePackageError,
    CheckpointBundleSourceArtifact,
    package_checkpoint_bundle,
)
from rl_fzerox.core.manager.store import ManagerStore, default_manager_db_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = ManagerStore(args.db)
    try:
        result = package_checkpoint_bundle(
            store=store,
            run_id=args.run_id,
            artifact=_artifact(args.artifact),
            version=args.version,
            checkpoint_id=args.checkpoint_id,
            checkpoint_name=args.name,
            output_path=args.output,
            allow_running=args.allow_running,
            overwrite=args.overwrite,
        )
    except CheckpointBundlePackageError as exc:
        raise SystemExit(str(exc)) from exc

    print(result.bundle_path)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package one local manager checkpoint as a release bundle."
    )
    parser.add_argument("--db", type=Path, default=default_manager_db_path())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", choices=("latest", "best", "final"), default="best")
    parser.add_argument("--version", required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _artifact(value: str) -> CheckpointBundleSourceArtifact:
    match value:
        case "latest":
            return "latest"
        case "best":
            return "best"
        case "final":
            return "final"
        case _:
            raise CheckpointBundlePackageError(f"unsupported artifact: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
