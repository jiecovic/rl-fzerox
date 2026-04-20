# src/rl_fzerox/apps/recording/runner.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.recording.models import (
    AttemptRunResult,
    RecordAttemptResult,
    RecordingSession,
    RecordMode,
)
from rl_fzerox.apps.recording.progress import (
    ProgressPrinter,
    format_race_time_ms,
    int_info,
    optional_str_info,
)
from rl_fzerox.apps.recording.session import open_recording_session
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    VideoSettings,
    as_rgb_frame,
    attempt_output_path,
    discard_attempt_video,
    ensure_attempt_path_available,
    resolve_ffmpeg_path,
    temp_session_id,
)
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.training.inference import PolicyRunner


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
    session = open_recording_session(config, fps=fps)
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
                watch_step = env.step_watch(action)
                observation = watch_step.observation
                reward = watch_step.reward
                terminated = watch_step.terminated
                truncated = watch_step.truncated
                info = watch_step.info
                episode_return += reward
                progress.print(info, episode_return=episode_return)
                display_frames = watch_step.display_frames or (env.render(),)
                for display_frame in display_frames:
                    writer.write(as_rgb_frame(display_frame))

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
