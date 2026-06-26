# scripts/import_checkpoint_bundle.py
"""Maintainer/local tool for validating and installing a checkpoint bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_fzerox.core.manager.checkpoints import (  # noqa: E402
    CheckpointBundleImportError,
)
from rl_fzerox.core.manager.store import ManagerStore, default_manager_db_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = ManagerStore(args.db)
    try:
        checkpoint = store.import_published_checkpoint_bundle(
            bundle_path=args.bundle,
            target_root=args.target_root,
        )
    except (CheckpointBundleImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"{checkpoint.id} {checkpoint.import_dir}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and install one checkpoint release bundle locally."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--db", type=Path, default=default_manager_db_path())
    parser.add_argument("--target-root", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
