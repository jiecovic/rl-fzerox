# src/rl_fzerox/apps/telemetry.py
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.boot import boot_into_first_race
from rl_fzerox.core.config.loader import load_watch_app_config
from rl_fzerox.core.emulator import Emulator
from rl_fzerox.core.game import read_telemetry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the live telemetry probe."""

    parser = argparse.ArgumentParser(
        description="Read live F-Zero X telemetry from emulated system RAM.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        required=True,
        type=Path,
        help="Path to the watch config YAML.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Additional frames to advance after reset/bootstrap.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Open the emulator from config and print a telemetry JSON snapshot."""

    args = parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must be >= 0")

    config = load_watch_app_config(args.config_path)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
    )

    try:
        reset_state = emulator.reset()
        boot_info: dict[str, object] = {}
        if config.env.reset_to_race and emulator.baseline_kind != "custom":
            _, boot_info = boot_into_first_race(emulator)
        if args.frames:
            emulator.step_frames(args.frames)

        telemetry = read_telemetry(emulator)
        print(
            json.dumps(
                {
                    "frame_index": emulator.frame_index,
                    "baseline_kind": emulator.baseline_kind,
                    "reset_mode": boot_info.get("reset_mode", "baseline"),
                    "boot_state": boot_info.get("boot_state", "-"),
                    "reset_frame_shape": list(reset_state.frame.shape),
                    "telemetry": telemetry.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        emulator.close()


if __name__ == "__main__":
    main()
