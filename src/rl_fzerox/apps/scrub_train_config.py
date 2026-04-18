# src/rl_fzerox/apps/scrub_train_config.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.training.runs import scrub_obsolete_train_run_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments for explicit saved train-config maintenance."""

    parser = argparse.ArgumentParser(
        description="Scrub known obsolete fields from a saved train_config.yaml.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Training run directory containing train_config.yaml.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=None,
        help="Write the scrubbed config to this path instead of editing in place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite train_config.yaml in the run directory.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create train_config.yaml.bak when using --in-place.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the manifest scrub command."""

    args = parse_args(argv)
    try:
        result = scrub_obsolete_train_run_config(
            args.run_dir,
            output_path=args.output_path,
            in_place=args.in_place,
            backup=not args.no_backup,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if result.output_path is None:
        print(f"dry run: {result.source_path}")
    else:
        print(f"wrote {result.output_path}")
    if result.backup_path is not None:
        print(f"backup {result.backup_path}")
    if result.removed_fields:
        print("removed:")
        for field in result.removed_fields:
            print(f"  - {field}")
    else:
        print("removed: none")
    if result.output_path is None:
        print("use --in-place or --out to write the scrubbed config")


if __name__ == "__main__":
    main()
