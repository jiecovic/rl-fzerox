# src/rl_fzerox/apps/bench_renderer.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from fzerox_emulator import JOYPAD_A, ControllerState, Emulator, joypad_mask
from fzerox_emulator.base import BackendStepResult

RENDERERS: tuple[str, ...] = ("angrylion", "gliden64")


@dataclass(frozen=True, slots=True)
class RendererBenchResult:
    """Single-renderer benchmark result printed by the CLI."""

    renderer: str
    ok: bool
    frames: int
    seconds: float
    frames_per_second: float
    frame_shape: tuple[int, int, int] | None = None
    error: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse renderer benchmark CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark libretro renderer backends through the native env-step path.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--core-path",
        type=Path,
        default=Path("local/libretro/mupen64plus_next_libretro.so"),
        help="Path to the Mupen64Plus-Next libretro core.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=Path("local/roms/F-Zero X (USA).n64"),
        help="Path to the F-Zero X ROM.",
    )
    parser.add_argument(
        "--baseline-state",
        type=Path,
        default=Path("local/states/first-race.state"),
        help="Race-start savestate used for reset.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("local/runtime/renderer-bench"),
        help="Runtime directory for core files generated during the benchmark.",
    )
    parser.add_argument(
        "--renderer",
        action="append",
        choices=RENDERERS,
        help="Renderer to test. Repeat to test multiple; defaults to all known renderers.",
    )
    parser.add_argument("--frames", type=int, default=1_000, help="Measured env steps.")
    parser.add_argument("--warmup", type=int, default=120, help="Warmup env steps.")
    parser.add_argument("--preset", default="crop_116x164", help="Observation preset.")
    parser.add_argument("--frame-stack", type=int, default=4, help="Observation frame stack.")
    parser.add_argument("--action-repeat", type=int, default=1, help="Internal frames per step.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a compact table.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run renderers in this process. Default isolates each renderer subprocess.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the renderer benchmark."""

    args = parse_args(argv)
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.action_repeat <= 0:
        raise SystemExit("--action-repeat must be > 0")
    if args.frame_stack <= 0:
        raise SystemExit("--frame-stack must be > 0")

    renderers = tuple(args.renderer) if args.renderer is not None else RENDERERS
    results = (
        tuple(_benchmark_renderer(args, renderer) for renderer in renderers)
        if args.in_process
        else tuple(_benchmark_renderer_isolated(args, renderer) for renderer in renderers)
    )
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_table(results)


def _benchmark_renderer(args: argparse.Namespace, renderer: str) -> RendererBenchResult:
    runtime_dir = (args.runtime_dir / renderer).expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    controller = ControllerState(joypad_mask=joypad_mask(JOYPAD_A))

    try:
        emulator = Emulator(
            core_path=args.core_path.expanduser().resolve(),
            rom_path=args.rom_path.expanduser().resolve(),
            runtime_dir=runtime_dir,
            baseline_state_path=args.baseline_state.expanduser().resolve(),
            renderer=renderer,
        )
    except Exception as error:
        return _failed_result(renderer, str(error))

    frames_run = 0
    try:
        emulator.reset()
        for _ in range(args.warmup):
            _step(emulator, controller, args)
        start = time.perf_counter()
        for _ in range(args.frames):
            result = _step(emulator, controller, args)
            frames_run += int(result.summary.frames_run)
        elapsed = time.perf_counter() - start
        fps = frames_run / elapsed if elapsed > 0 else 0.0
        return RendererBenchResult(
            renderer=renderer,
            ok=True,
            frames=frames_run,
            seconds=elapsed,
            frames_per_second=fps,
            frame_shape=emulator.frame_shape,
        )
    except Exception as error:
        return _failed_result(renderer, str(error), frames=frames_run)
    finally:
        emulator.close()


def _benchmark_renderer_isolated(args: argparse.Namespace, renderer: str) -> RendererBenchResult:
    command = [
        sys.executable,
        "-m",
        "rl_fzerox.apps.bench_renderer",
        "--renderer",
        renderer,
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--preset",
        str(args.preset),
        "--frame-stack",
        str(args.frame_stack),
        "--action-repeat",
        str(args.action_repeat),
        "--core-path",
        str(args.core_path),
        "--rom-path",
        str(args.rom_path),
        "--baseline-state",
        str(args.baseline_state),
        "--runtime-dir",
        str(args.runtime_dir),
        "--json",
        "--in-process",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        if not error:
            error = _return_code_message(completed.returncode)
        return _failed_result(renderer, error)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return _failed_result(renderer, f"invalid benchmark JSON: {error}")
    if not isinstance(payload, list) or not payload:
        return _failed_result(renderer, "benchmark subprocess returned no result")
    return _result_from_payload(renderer, payload[0])


def _step(
    emulator: Emulator, controller: ControllerState, args: argparse.Namespace
) -> BackendStepResult:
    return emulator.step_repeat_raw(
        controller_state=controller,
        action_repeat=args.action_repeat,
        preset=args.preset,
        frame_stack=args.frame_stack,
        stuck_min_speed_kph=60.0,
        energy_loss_epsilon=0.01,
        max_episode_steps=100_000,
        progress_frontier_stall_limit_frames=None,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=False,
    )


def _result_from_payload(renderer: str, payload: object) -> RendererBenchResult:
    if not isinstance(payload, dict):
        return _failed_result(renderer, "benchmark subprocess returned malformed result")
    return RendererBenchResult(
        renderer=str(payload.get("renderer", renderer)),
        ok=bool(payload.get("ok", False)),
        frames=int(payload.get("frames", 0)),
        seconds=float(payload.get("seconds", 0.0)),
        frames_per_second=float(payload.get("frames_per_second", 0.0)),
        frame_shape=_frame_shape_from_payload(payload.get("frame_shape")),
        error=None if payload.get("error") is None else str(payload["error"]),
    )


def _frame_shape_from_payload(payload: object) -> tuple[int, int, int] | None:
    if not isinstance(payload, list | tuple) or len(payload) != 3:
        return None
    return int(payload[0]), int(payload[1]), int(payload[2])


def _return_code_message(return_code: int) -> str:
    if return_code < 0:
        return f"process terminated by signal {-return_code}"
    return f"process exited with code {return_code}"


def _failed_result(renderer: str, error: str, *, frames: int = 0) -> RendererBenchResult:
    return RendererBenchResult(
        renderer=renderer,
        ok=False,
        frames=frames,
        seconds=0.0,
        frames_per_second=0.0,
        error=error,
    )


def _print_table(results: tuple[RendererBenchResult, ...]) -> None:
    for result in results:
        if result.ok:
            shape = "x".join(str(value) for value in result.frame_shape or ())
            print(
                f"{result.renderer:12s} ok    "
                f"{result.frames:7d} frames  "
                f"{result.frames_per_second:7.1f} fps  "
                f"{result.seconds:7.2f}s  "
                f"shape={shape}"
            )
        else:
            print(f"{result.renderer:12s} fail  {result.error}")


if __name__ == "__main__":
    main()
