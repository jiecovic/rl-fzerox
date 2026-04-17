# src/rl_fzerox/apps/record_policy.py
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from fzerox_emulator import Emulator
from rl_fzerox.apps.watch import resolve_watch_app_config
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.inference import PolicyRunner, load_policy_runner

RgbFrame = NDArray[np.uint8]


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
class VideoSettings:
    """MP4 writer settings for one attempt."""

    path: Path
    ffmpeg_path: str
    fps: float


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
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "saved "
        f"{result.path} "
        f"(attempt={result.attempt}, rank={result.finish_rank}, "
        f"time={_format_race_time_ms(result.race_time_ms)}, "
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

    ffmpeg_path = _resolve_ffmpeg_path()
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
) -> RecordAttemptResult:
    session = _open_recording_session(config, fps=fps)
    temp_session_id = _temp_session_id()
    try:
        for attempt in range(1, attempts + 1):
            attempt_path = _attempt_output_path(
                output_path,
                attempt,
                session_id=temp_session_id,
            )
            _ensure_attempt_path_available(attempt_path)
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
                ),
                progress_interval_seconds=progress_interval_seconds,
            )
            _print_attempt_result(result)
            if result.matched:
                return _move_result_to_output(result, output_path)
            if not keep_failed:
                _discard_attempt_video(result.path)

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
    output_fps = _resolve_video_fps(
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
    video: VideoSettings,
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

    with FfmpegRgbWriter(
        path=video.path,
        ffmpeg_path=video.ffmpeg_path,
        fps=video.fps,
    ) as writer:
        initial_frame = _as_rgb_frame(env.render())
        writer.write(initial_frame)
        while not (terminated or truncated):
            action = policy_runner.predict(
                observation,
                deterministic=deterministic,
                action_masks=env.action_masks(),
            )
            control_state = env.action_to_control_state(action)
            observation, reward, terminated, truncated, info = env.step_control(control_state)
            episode_return += reward
            progress.print(info, episode_return=episode_return)
            writer.write(_as_rgb_frame(env.render()))

    progress.finish()
    finish_rank = _finished_rank(info)
    matched = finish_rank is not None and finish_rank <= target_rank
    return AttemptRunResult(
        attempt=attempt,
        path=path,
        matched=matched,
        finish_rank=finish_rank,
        episode_return=episode_return,
        episode_steps=_int_info(info, "episode_step"),
        race_time_ms=_int_info(info, "race_time_ms"),
        termination_reason=_optional_str_info(info, "termination_reason"),
        truncation_reason=_optional_str_info(info, "truncation_reason"),
    )


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


class FfmpegRgbWriter:
    """Encode one attempt by streaming rendered RGB frames into ffmpeg."""

    def __init__(
        self,
        *,
        path: Path,
        ffmpeg_path: str,
        fps: float,
    ) -> None:
        self._path = path
        self._ffmpeg_path = ffmpeg_path
        self._fps = fps
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._shape: tuple[int, int, int] | None = None

    def __enter__(self) -> FfmpegRgbWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, frame: RgbFrame) -> None:
        copied_frame = np.ascontiguousarray(frame)
        if self._shape is None:
            if copied_frame.ndim != 3 or copied_frame.shape[2] != 3:
                raise ValueError(f"expected RGB frame with shape HxWx3, got {copied_frame.shape}")
            self._shape = (
                int(copied_frame.shape[0]),
                int(copied_frame.shape[1]),
                int(copied_frame.shape[2]),
            )
            self._open_process(width=self._shape[1], height=self._shape[0])
        if copied_frame.shape != self._shape:
            raise ValueError(
                f"frame shape changed during recording: {copied_frame.shape} != {self._shape}"
            )
        if self._stdin is None:
            raise RuntimeError("ffmpeg writer is not open")
        self._stdin.write(copied_frame.tobytes())

    def _open_process(self, *, width: int, height: int) -> None:
        command = _ffmpeg_command(
            ffmpeg_path=self._ffmpeg_path,
            output_path=self._path,
            width=width,
            height=height,
            fps=self._fps,
        )
        self._process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")
        self._stdin = self._process.stdin

    def close(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None
        if self._process is None:
            return
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            detail = stderr.decode(errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}{suffix}")


def _discard_attempt_video(path: Path) -> None:
    path.unlink(missing_ok=True)


class ProgressPrinter:
    """Render one compact live status line while an attempt is recording."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        attempt: int,
        target_rank: int,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._attempt = attempt
        self._target_rank = target_rank
        self._started_at = time.monotonic()
        self._next_print_time = 0.0
        self._printed = False

    def print(
        self,
        info: dict[str, object],
        *,
        episode_return: float,
        force: bool = False,
    ) -> None:
        if self._interval_seconds <= 0.0:
            return
        now = time.monotonic()
        if not force and now < self._next_print_time:
            return
        self._next_print_time = now + self._interval_seconds
        self._printed = True
        line = _format_progress_line(
            info,
            attempt=self._attempt,
            target_rank=self._target_rank,
            episode_return=episode_return,
            effective_fps=_effective_fps(info, started_at=self._started_at, now=now),
        )
        print(
            f"\r\x1b[2K{line}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        if self._printed:
            print()


def _format_progress_line(
    info: dict[str, object],
    *,
    attempt: int,
    target_rank: int,
    episode_return: float,
    effective_fps: float,
) -> str:
    return (
        f"try {attempt:02d} | "
        f"step {_int_info(info, 'episode_step')} | "
        f"rank {_int_info(info, 'position')} | "
        f"lap {_int_info(info, 'lap')} | "
        f"time {_format_race_time_ms(_int_info(info, 'race_time_ms'))} | "
        f"{_float_info(info, 'speed_kph'):.0f} km/h | "
        f"{_format_compact_number(_float_info(info, 'race_distance'))} prog | "
        f"{effective_fps:.1f} frames/s | "
        f"R {episode_return:.1f} | "
        f"{_format_target_rank(target_rank)}"
    )


def _ffmpeg_command(
    *,
    ffmpeg_path: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _resolve_ffmpeg_path() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    bundled_ffmpeg = _imageio_ffmpeg_path()
    if bundled_ffmpeg is not None:
        return bundled_ffmpeg
    raise RuntimeError(
        "ffmpeg is required for MP4 recording. Install system ffmpeg or run "
        "`.venv/bin/python -m pip install imageio-ffmpeg`."
    )


def _imageio_ffmpeg_path() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()


def _with_deterministic_policy(
    config: WatchAppConfig,
    *,
    deterministic: bool,
) -> WatchAppConfig:
    return config.model_copy(
        update={"watch": config.watch.model_copy(update={"deterministic_policy": deterministic})}
    )


def _resolve_video_fps(
    *,
    native_fps: float,
    action_repeat: int,
    override: float | None,
) -> float:
    if override is not None:
        return override
    return max(float(native_fps) / float(action_repeat), 1.0)


def _finished_rank(info: dict[str, object]) -> int | None:
    if info.get("termination_reason") != "finished":
        return None
    rank = info.get("position")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        return None
    return rank


def _attempt_output_path(
    output_path: Path,
    attempt: int,
    *,
    session_id: str | None = None,
) -> Path:
    if session_id is None:
        return output_path.with_name(f".{output_path.stem}.attempt-{attempt:03d}.mp4")
    return output_path.with_name(f".{output_path.stem}.{session_id}.attempt-{attempt:03d}.mp4")


def _temp_session_id() -> str:
    return f"session-{os.getpid()}-{time.time_ns()}"


def _ensure_attempt_path_available(path: Path) -> None:
    if not path.exists():
        return
    raise FileExistsError(f"temporary recording artifact already exists: {path}")


def _as_rgb_frame(frame: np.ndarray) -> RgbFrame:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape HxWx3, got {frame.shape}")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _int_info(info: dict[str, object], key: str) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _float_info(info: dict[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _format_compact_number(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.2f}M"
    if absolute_value >= 10_000.0:
        return f"{value / 1_000.0:.1f}k"
    if absolute_value >= 1_000.0:
        return f"{value / 1_000.0:.2f}k"
    return f"{value:.0f}"


def _format_target_rank(target_rank: int) -> str:
    if target_rank == 1:
        return "need rank 1"
    return f"need rank <= {target_rank}"


def _format_race_time_ms(milliseconds: int) -> str:
    if milliseconds <= 0:
        return "--:--.---"
    minutes, remaining_ms = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remaining_ms, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _effective_fps(
    info: dict[str, object],
    *,
    started_at: float,
    now: float,
) -> float:
    elapsed = max(now - started_at, 1e-9)
    return float(_int_info(info, "episode_step")) / elapsed


def _optional_str_info(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    return value


def _print_attempt_result(result: RecordAttemptResult | AttemptRunResult) -> None:
    reason = result.termination_reason or result.truncation_reason or "unknown"
    rank = "-" if result.finish_rank is None else str(result.finish_rank)
    status = "keep" if result.matched else "discard"
    print(
        f"attempt {result.attempt}: {status}, reason={reason}, rank={rank}, "
        f"time={_format_race_time_ms(result.race_time_ms)}, "
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
