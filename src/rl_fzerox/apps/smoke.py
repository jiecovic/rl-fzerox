# src/rl_fzerox/apps/smoke.py
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.emulator import Emulator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the headless smoke test."""

    parser = argparse.ArgumentParser(
        description="Load the ROM, reset the emulator, and advance frames headlessly.",
        allow_abbrev=False,
    )
    parser.add_argument("core_path", help="Path to a libretro core shared library.")
    parser.add_argument("rom_path", help="Path to the F-Zero X ROM.")
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to advance after reset.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the headless ROM-load and frame-step smoke path."""

    args = parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must be >= 0")

    emulator = Emulator(
        core_path=Path(args.core_path).expanduser().resolve(),
        rom_path=Path(args.rom_path).expanduser().resolve(),
    )
    try:
        reset_state = emulator.reset()
        emulator.step_frames(args.frames)
        frame = emulator.render()
        print(
            json.dumps(
                {
                    "backend": emulator.name,
                    "display_aspect_ratio": emulator.display_aspect_ratio,
                    "display_size": list(emulator.display_size),
                    "native_fps": emulator.native_fps,
                    "frame_shape": list(emulator.frame_shape),
                    "frame_index": emulator.frame_index,
                    "reset_frame_shape": list(reset_state.frame.shape),
                    "final_frame_shape": list(frame.shape),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        emulator.close()


if __name__ == "__main__":
    main()
