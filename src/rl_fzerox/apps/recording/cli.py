# src/rl_fzerox/apps/recording/cli.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.apps.recording.progress import format_race_time_ms
from rl_fzerox.apps.recording.runner import record_policy_episode
from rl_fzerox.apps.watch import resolve_watch_app_config
from rl_fzerox.core.config.schema import WatchAppConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments for conditional headless policy recording."""

    parser = argparse.ArgumentParser(
        description="Record a policy episode to MP4 only when it matches a finish condition.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to a watch config YAML file.",
    )
    parser.add_argument(
        "--run-dir",
        dest="policy_run_dir",
        type=Path,
        default=None,
        help="Training run directory containing saved policy artifacts.",
    )
    parser.add_argument(
        "--artifact",
        dest="policy_artifact",
        choices=("latest", "best", "final"),
        default=None,
        help="Which saved policy artifact to load from the run directory.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        type=Path,
        help="Final MP4 path to create when a matching episode is found.",
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=50,
        help="Maximum number of attempts before giving up.",
    )
    parser.add_argument(
        "--target-rank",
        type=_positive_int,
        default=1,
        help="Keep the first finished episode with rank <= this value.",
    )
    parser.add_argument(
        "--fps",
        type=_positive_float,
        default=None,
        help="Output video FPS. Defaults to the emulator native FPS.",
    )
    parser.add_argument(
        "--progress-interval",
        type=_non_negative_float,
        default=2.0,
        help="Seconds between live terminal progress updates. Use 0 to disable.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions. Pass --no-deterministic for stochastic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --out if it already exists.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep temporary MP4s for attempts that do not match the condition.",
    )
    parser.add_argument(
        "--record-mode",
        choices=("stream-all", "probe-then-record"),
        default="stream-all",
        help=(
            "stream-all records every attempt as it runs; probe-then-record skips "
            "failed video encoding and replays the first matching attempt."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra watch overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run conditional policy recording from the CLI."""

    args = parse_args(argv)
    try:
        config = resolve_watch_app_config(
            config_path=args.config_path,
            policy_run_dir=args.policy_run_dir,
            policy_artifact=args.policy_artifact,
            overrides=args.overrides,
        )
        config = with_deterministic_policy(config, deterministic=args.deterministic)
        output_path = args.output_path.expanduser().resolve()
        result = record_policy_episode(
            config,
            output_path=output_path,
            attempts=args.episodes,
            target_rank=args.target_rank,
            fps=args.fps,
            progress_interval_seconds=args.progress_interval,
            overwrite=args.overwrite,
            keep_failed=args.keep_failed,
            record_mode=args.record_mode,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "saved "
        f"{result.path} "
        f"(attempt={result.attempt}, rank={result.finish_rank}, "
        f"time={format_race_time_ms(result.race_time_ms)}, "
        f"return={result.episode_return:.3f}, steps={result.episode_steps})"
    )


def with_deterministic_policy(
    config: WatchAppConfig,
    *,
    deterministic: bool,
) -> WatchAppConfig:
    return config.model_copy(
        update={"watch": config.watch.model_copy(update={"deterministic_policy": deterministic})}
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed
