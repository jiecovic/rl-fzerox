# src/rl_fzerox/apps/_cli.py
from __future__ import annotations

from collections.abc import Sequence


def normalize_cli_overrides(overrides: Sequence[str]) -> list[str]:
    """Normalize argparse remainder overrides for dotlist-style CLIs."""

    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)
