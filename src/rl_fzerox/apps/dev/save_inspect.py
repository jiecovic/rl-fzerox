# src/rl_fzerox/apps/dev/save_inspect.py
"""Inspect a local F-Zero X save-RAM file without interpreting unlock offsets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rl_fzerox.core.save_game import summarize_save_ram


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a conservative summary of a portable F-Zero X save-RAM file.",
    )
    parser.add_argument("path", type=Path, help="Path to the save-RAM file, normally *.srm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = args.path.read_bytes()
    print(json.dumps(asdict(summarize_save_ram(data)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
