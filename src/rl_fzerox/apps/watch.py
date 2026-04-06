# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.config import default_config_dir, load_watch_app_config
from rl_fzerox.ui import run_viewer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment with Hydra-style config overrides.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-cfg",
        "--config-file",
        dest="config_file",
        default=None,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "-c",
        "--config-name",
        dest="config_name",
        default="watch",
        help="Config name without the .yaml suffix.",
    )
    parser.add_argument(
        "-p",
        "--config-path",
        dest="config_path",
        default=str(default_config_dir()),
        help="Hydra config directory.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir, config_name = _resolve_config_selection(args)
    try:
        config = load_watch_app_config(
            config_name=config_name,
            config_dir=config_dir,
            overrides=_normalize_overrides(args.overrides),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_viewer(config)


def _resolve_config_selection(args: argparse.Namespace) -> tuple[Path, str]:
    if args.config_file:
        config_file = Path(str(args.config_file)).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"--config-file not found: {config_file}")
        return config_file.parent, config_file.stem

    config_path = Path(str(args.config_path)).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    return config_path, str(args.config_name)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


if __name__ == "__main__":
    main()
