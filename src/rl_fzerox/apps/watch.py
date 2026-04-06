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
        description="Watch the F-Zero X environment from a YAML config file.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        required=True,
        help="Path to a watch config YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    try:
        config = load_watch_app_config(Path(str(args.config_path)))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_viewer(config)


if __name__ == "__main__":
    main()
