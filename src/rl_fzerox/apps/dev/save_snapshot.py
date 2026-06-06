# src/rl_fzerox/apps/dev/save_snapshot.py
"""Capture portable save RAM from a headless emulator session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from fzerox_emulator import Emulator
from rl_fzerox.core.manager.projection.assembly import default_core_path, default_rom_path
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, KNOWN_RENDERERS
from rl_fzerox.core.save_game import summarize_save_ram


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the save-RAM snapshot probe."""

    parser = argparse.ArgumentParser(
        description="Boot the emulator and write the current portable F-Zero X save RAM.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--core",
        dest="core_path",
        type=Path,
        default=default_core_path(),
        help="Path to the libretro core shared library.",
    )
    parser.add_argument(
        "--rom",
        dest="rom_path",
        type=Path,
        default=default_rom_path(),
        help="Path to the F-Zero X USA ROM.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Optional runtime directory for emulator-generated files.",
    )
    parser.add_argument(
        "--renderer",
        choices=KNOWN_RENDERERS,
        default=DEFAULT_RENDERER,
        help="Retro plugin renderer to request from the core.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional save-RAM file to inject before stepping.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frames to advance after optional save injection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output path if it already exists.",
    )
    parser.add_argument("output", type=Path, help="Path to write the captured save-RAM bytes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture save RAM from one emulator process."""

    args = parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must be >= 0")

    output_path = _resolved_path(args.output)
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists, pass --force to overwrite: {output_path}")

    emulator = Emulator(
        core_path=_resolved_path(args.core_path),
        rom_path=_resolved_path(args.rom_path),
        runtime_dir=None if args.runtime_dir is None else _resolved_path(args.runtime_dir),
        renderer=args.renderer,
    )
    try:
        if args.input is not None:
            emulator.write_save_ram(_resolved_path(args.input).read_bytes())
        if args.frames:
            emulator.step_frames(args.frames, capture_video=False)

        data = emulator.read_save_ram()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "renderer": args.renderer,
                    "frame_index": emulator.frame_index,
                    "save_ram": asdict(summarize_save_ram(data)),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        emulator.close()

    return 0


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
