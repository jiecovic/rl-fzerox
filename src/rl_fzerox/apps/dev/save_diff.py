# src/rl_fzerox/apps/dev/save_diff.py
"""Diff two local F-Zero X save-RAM files at byte-range level."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rl_fzerox.core.save_game import diff_save_ram, diff_save_ram_bits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print coalesced changed byte ranges between two F-Zero X save-RAM files.",
    )
    parser.add_argument("before", type=Path, help="Path to the earlier save-RAM file")
    parser.add_argument("after", type=Path, help="Path to the later save-RAM file")
    parser.add_argument(
        "--max-ranges",
        type=int,
        default=64,
        help="Maximum changed ranges to print; changed byte count is always complete",
    )
    parser.add_argument(
        "--bits",
        action="store_true",
        help="Print changed bits instead of coalesced byte ranges",
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=256,
        help="Maximum changed bits to print when --bits is set; changed bit count is complete",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    before = args.before.read_bytes()
    after = args.after.read_bytes()
    diff = (
        diff_save_ram_bits(before, after, max_bits=args.max_bits)
        if args.bits
        else diff_save_ram(before, after, max_ranges=args.max_ranges)
    )
    print(json.dumps(asdict(diff), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
