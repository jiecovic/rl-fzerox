# src/rl_fzerox/apps/record_policy.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fzerox_emulator import Emulator
from rl_fzerox.apps.recording.progress import (
    ProgressPrinter,
    format_race_time_ms,
    int_info,
    optional_str_info,
)
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    VideoSettings,
    as_rgb_frame,
    attempt_output_path,
    discard_attempt_video,
    ensure_attempt_path_available,
    resolve_ffmpeg_path,
    resolve_video_fps,
    temp_session_id,
)
from rl_fzerox.apps.watch import resolve_watch_app_config
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.inference import PolicyRunner, load_policy_runner

RecordMode = Literal["stream-all", "probe-then-record"]


@dataclass(frozen=True)
class RecordAttemptResult:
    """Outcome from one recorded policy episode."""

    attempt: int
    path: Path
    matched: bool
    finish_rank: int | None
    episode_return: float
    episode_steps: int
    race_time_ms: int
    termination_reason: str | None
    truncation_reason: str | None


@dataclass(frozen=True)
class AttemptRunResult:
    """Internal attempt result from one temp-recorded episode."""

    attempt: int
    path: Path
    matched: bool
    finish_rank: int | None
    episode_return: float
    episode_steps: int
    race_time_ms: int
    termination_reason: str | None
    truncation_reason: str | None


@dataclass(frozen=True)
class RecordingSession:
    """Open emulator env plus policy runner for one recording/probing pass."""

    env: FZeroXEnv
    policy_runner: PolicyRunner
    output_fps: float


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
        help="Output video FPS. Defaults to native_fps / action_repeat.",
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
        config = _with_deterministic_policy(config, deterministic=args.deterministic)
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


def record_policy_episode(
    config: WatchAppConfig,
    *,
    output_path: Path,
    attempts: int,
    target_rank: int,
    fps: float | None = None,
    progress_interval_seconds: float = 2.0,
    overwrite: bool = False,
    keep_failed: bool = False,
    record_mode: RecordMode = "stream-all",
) -> RecordAttemptResult:
    """Record attempts until a finished episode satisfies the target rank."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if target_rank <= 0:
        raise ValueError("target_rank must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if config.watch.policy_run_dir is None:
        raise ValueError("--run-dir or watch.policy_run_dir is required for policy recording")
    if keep_failed and record_mode != "stream-all":
        raise ValueError("--keep-failed requires --record-mode stream-all")

    ffmpeg_path = resolve_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return _record_attempts(
        config,
        output_path=output_path,
        attempts=attempts,
        target_rank=target_rank,
        ffmpeg_path=ffmpeg_path,
        fps=fps,
        progress_interval_seconds=progress_interval_seconds,
        keep_failed=keep_failed,
        record_mode=record_mode,
    )


def _record_attempts(
    config: WatchAppConfig,
    *,
    output_path: Path,
    attempts: int,
    target_rank: int,
    ffmpeg_path: str,
    fps: float | None,
    progress_interval_seconds: float,
    keep_failed: bool,
    record_mode: RecordMode,
) -> RecordAttemptResult:
    session = _open_recording_session(config, fps=fps)
    session_id = temp_session_id()
    try:
        for attempt in range(1, attempts + 1):
            attempt_path = attempt_output_path(
                output_path,
                attempt,
                session_id=session_id,
            )
            ensure_attempt_path_available(attempt_path)
            result = _run_attempt(
                session.env,
                policy_runner=session.policy_runner,
                path=attempt_path,
                attempt=attempt,
                seed=_attempt_seed(config.seed, attempt),
                deterministic=config.watch.deterministic_policy,
                target_rank=target_rank,
                video=VideoSettings(
                    path=attempt_path,
                    ffmpeg_path=ffmpeg_path,
                    fps=session.output_fps,
                )
                if record_mode == "stream-all"
                else None,
                progress_interval_seconds=progress_interval_seconds,
            )
            _print_attempt_result(result)
            if result.matched:
                if record_mode == "probe-then-record":
                    result = _record_matched_attempt(
                        session,
                        attempt_path=attempt_path,
                        attempt=attempt,
                        seed=_attempt_seed(config.seed, attempt),
                        deterministic=config.watch.deterministic_policy,
                        target_rank=target_rank,
                        video=VideoSettings(
                            path=attempt_path,
                            ffmpeg_path=ffmpeg_path,
                            fps=session.output_fps,
                        ),
                        progress_interval_seconds=progress_interval_seconds,
                    )
                return _move_result_to_output(result, output_path)
            if not keep_failed:
                discard_attempt_video(result.path)

        raise RuntimeError(
            f"No finished episode reached rank <= {target_rank} after {attempts} attempts"
        )
    finally:
        session.env.close()


def _open_recording_session(
    config: WatchAppConfig,
    *,
    fps: float | None,
) -> RecordingSession:
    if config.watch.policy_run_dir is None:
        raise ValueError("--run-dir or watch.policy_run_dir is required for policy recording")
    seed_process(config.seed)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    policy_runner = load_policy_runner(
        config.watch.policy_run_dir,
        artifact=config.watch.policy_artifact,
        device=config.watch.device,
    )
    env.sync_checkpoint_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)
    output_fps = resolve_video_fps(
        native_fps=env.backend.native_fps,
        action_repeat=config.env.action_repeat,
        override=fps,
    )
    return RecordingSession(env=env, policy_runner=policy_runner, output_fps=output_fps)


def _run_attempt(
    env: FZeroXEnv,
    *,
    policy_runner: PolicyRunner,
    path: Path,
    attempt: int,
    seed: int | None,
    deterministic: bool,
    target_rank: int,
    video: VideoSettings | None,
    progress_interval_seconds: float,
) -> AttemptRunResult:
    observation, info = env.reset(seed=seed)
    policy_runner.reset()
    env.sync_checkpoint_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)
    episode_return = 0.0
    terminated = False
    truncated = False
    progress = ProgressPrinter(
        interval_seconds=progress_interval_seconds,
        attempt=attempt,
        target_rank=target_rank,
    )
    progress.print(info, episode_return=episode_return, force=True)

    if video is None:
        while not (terminated or truncated):
            action = policy_runner.predict(
                observation,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            progress.print(info, episode_return=episode_return)
    else:
        with FfmpegRgbWriter(
            path=video.path,
            ffmpeg_path=video.ffmpeg_path,
            fps=video.fps,
        ) as writer:
            initial_frame = as_rgb_frame(env.render())
            writer.write(initial_frame)
            while not (terminated or truncated):
                action = policy_runner.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                progress.print(info, episode_return=episode_return)
                writer.write(as_rgb_frame(env.render()))

    progress.finish()
    finish_rank = _finished_rank(info)
    matched = finish_rank is not None and finish_rank <= target_rank
    return AttemptRunResult(
        attempt=attempt,
        path=path,
        matched=matched,
        finish_rank=finish_rank,
        episode_return=episode_return,
        episode_steps=int_info(info, "episode_step"),
        race_time_ms=int_info(info, "race_time_ms"),
        termination_reason=optional_str_info(info, "termination_reason"),
        truncation_reason=optional_str_info(info, "truncation_reason"),
    )


def _record_matched_attempt(
    session: RecordingSession,
    *,
    attempt_path: Path,
    attempt: int,
    seed: int | None,
    deterministic: bool,
    target_rank: int,
    video: VideoSettings,
    progress_interval_seconds: float,
) -> AttemptRunResult:
    result = _run_attempt(
        session.env,
        policy_runner=session.policy_runner,
        path=attempt_path,
        attempt=attempt,
        seed=seed,
        deterministic=deterministic,
        target_rank=target_rank,
        video=video,
        progress_interval_seconds=progress_interval_seconds,
    )
    if not result.matched:
        raise RuntimeError(
            "Matched probe attempt did not reproduce while recording. "
            "Use --record-mode stream-all for non-deterministic runs."
        )
    return result


def _move_result_to_output(
    result: AttemptRunResult,
    output_path: Path,
) -> RecordAttemptResult:
    if output_path.exists():
        output_path.unlink()
    result.path.replace(output_path)
    return RecordAttemptResult(
        attempt=result.attempt,
        path=output_path,
        matched=True,
        finish_rank=result.finish_rank,
        episode_return=result.episode_return,
        episode_steps=result.episode_steps,
        race_time_ms=result.race_time_ms,
        termination_reason=result.termination_reason,
        truncation_reason=result.truncation_reason,
    )


def _attempt_seed(base_seed: int | None, attempt: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + attempt - 1


def _with_deterministic_policy(
    config: WatchAppConfig,
    *,
    deterministic: bool,
) -> WatchAppConfig:
    return config.model_copy(
        update={"watch": config.watch.model_copy(update={"deterministic_policy": deterministic})}
    )


def _finished_rank(info: dict[str, object]) -> int | None:
    if info.get("termination_reason") != "finished":
        return None
    rank = info.get("position")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        return None
    return rank


def _print_attempt_result(result: RecordAttemptResult | AttemptRunResult) -> None:
    reason = result.termination_reason or result.truncation_reason or "unknown"
    rank = "-" if result.finish_rank is None else str(result.finish_rank)
    status = "keep" if result.matched else "discard"
    print(
        f"attempt {result.attempt}: {status}, reason={reason}, rank={rank}, "
        f"time={format_race_time_ms(result.race_time_ms)}, "
        f"return={result.episode_return:.3f}, steps={result.episode_steps}"
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


if __name__ == "__main__":
    main()
