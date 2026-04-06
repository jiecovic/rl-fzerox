# src/rl_fzerox/apps/probe.py
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.emulator import probe_core


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the core probe command."""

    parser = argparse.ArgumentParser(
        description="Load a libretro core natively and print its system metadata.",
        allow_abbrev=False,
    )
    parser.add_argument("core_path", help="Path to a libretro core shared library.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the core probe CLI."""

    args = parse_args(argv)
    core_info = probe_core(str(Path(args.core_path).expanduser().resolve()))
    print(
        json.dumps(
            {
                "api_version": core_info.api_version,
                "library_name": core_info.library_name,
                "library_version": core_info.library_version,
                "valid_extensions": core_info.valid_extensions,
                "requires_full_path": core_info.requires_full_path,
                "blocks_extract": core_info.blocks_extract,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
