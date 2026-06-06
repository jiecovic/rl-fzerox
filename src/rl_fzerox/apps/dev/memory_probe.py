# src/rl_fzerox/apps/dev/memory_probe.py
"""Probe named save-RAM and live system-RAM slices from one emulator session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from fzerox_emulator import Emulator
from rl_fzerox.core.manager.projection.assembly import default_core_path, default_rom_path
from rl_fzerox.core.runtime_spec.renderers import DEFAULT_RENDERER, KNOWN_RENDERERS
from rl_fzerox.core.save_game import collect_memory_probe_report, parse_memory_probe_definition


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for save/system memory probes."""

    parser = argparse.ArgumentParser(
        description="Probe named F-Zero X save-RAM and system-RAM slices.",
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
        help="Optional save-RAM file to inject before probing.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frames to advance after optional save injection.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help=(
            "Probe definition in key=region:offset:length:format[:label] form. "
            "Regions are save_ram/system_ram; offsets accept decimal or 0x hex."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one emulator memory probe and print a JSON report."""

    args = parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must be >= 0")

    definitions = tuple(parse_memory_probe_definition(spec) for spec in args.probe)
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

        report = collect_memory_probe_report(
            definitions,
            read_save_ram=emulator.read_save_ram,
            read_system_ram=emulator.read_system_ram,
        )
        print(
            json.dumps(
                {
                    "renderer": args.renderer,
                    "frame_index": emulator.frame_index,
                    "probe_count": len(definitions),
                    "memory": asdict(report),
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
