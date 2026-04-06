# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.config import load_watch_app_config
from rl_fzerox.ui.viewer import run_viewer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the watch app."""

    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment from a Hydra-composed YAML config.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        required=True,
        type=Path,
        help="Path to a watch config YAML file.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    try:
        config = load_watch_app_config(
            args.config_path,
            overrides=_normalize_overrides(args.overrides),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_viewer(config)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


if __name__ == "__main__":
    main()
