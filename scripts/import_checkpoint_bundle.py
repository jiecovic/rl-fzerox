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
    default_imported_checkpoint_root,
    import_checkpoint_bundle,
)
from rl_fzerox.core.manager.store import default_manager_db_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    target_root = args.target_root or default_imported_checkpoint_root(db_path=args.db)
    try:
        result = import_checkpoint_bundle(
            bundle_path=args.bundle,
            target_root=target_root,
            overwrite=args.overwrite,
        )
    except CheckpointBundleImportError as exc:
        raise SystemExit(str(exc)) from exc

    print(result.import_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and install one checkpoint release bundle locally."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--db", type=Path, default=default_manager_db_path())
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
