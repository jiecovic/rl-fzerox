# src/rl_fzerox/apps/recording/runner.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.recording.models import (
    AttemptRunResult,
    PolicyReloadMode,
    RecordAttemptResult,
    RecordingSession,
    RecordMode,
)
from rl_fzerox.apps.recording.progress import (
    ProgressPrinter,
    format_race_time_ms,
    format_recording_target,
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
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.core.training.inference import PolicyRunner


def record_policy_episode(
    config: WatchAppConfig,
    *,
    output_path: Path,
    attempts: int,
    target_laps: int = 3,
    target_rank: int | None = None,
    course_id: str | None = None,
    fps: float | None = None,
    progress_interval_seconds: float = 2.0,
    overwrite: bool = False,
    keep_failed: bool = False,
    record_mode: RecordMode = "stream-all",
    reload_mode: PolicyReloadMode = "off",
    reload_interval_seconds: float = 10.0,
) -> RecordAttemptResult:
    """Record attempts until one episode satisfies the configured target."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if target_laps <= 0:
        raise ValueError("target_laps must be positive")
    if target_rank is not None and target_rank <= 0:
        raise ValueError("target_rank must be positive")
    if reload_interval_seconds < 0.0:
        raise ValueError("reload_interval_seconds must be non-negative")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if config.watch.policy_run_dir is None:
        raise ValueError("--run-dir or watch.policy_run_dir is required for policy recording")
    if keep_failed and record_mode != "stream-all":
        raise ValueError("--keep-failed requires --record-mode stream-all")
    locked_course_id = (
        None
        if course_id is None
        else _resolve_recording_course_id(config.env.track_sampling, course_id)
    )

    ffmpeg_path = resolve_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return _record_attempts(
        config,
        output_path=output_path,
        attempts=attempts,
        target_laps=target_laps,
        target_rank=target_rank,
        course_id=locked_course_id,
        ffmpeg_path=ffmpeg_path,
        fps=fps,
        progress_interval_seconds=progress_interval_seconds,
        keep_failed=keep_failed,
        record_mode=record_mode,
        reload_mode=reload_mode,
        reload_interval_seconds=reload_interval_seconds,
    )


def _record_attempts(
    config: WatchAppConfig,
    *,
    output_path: Path,
    attempts: int,
    target_laps: int,
    target_rank: int | None,
    course_id: str | None,
    ffmpeg_path: str,
    fps: float | None,
    progress_interval_seconds: float,
    keep_failed: bool,
    record_mode: RecordMode,
    reload_mode: PolicyReloadMode,
    reload_interval_seconds: float,
) -> RecordAttemptResult:
    session = open_recording_session(config, fps=fps)
    if course_id is not None:
        session.env.set_locked_reset_course(course_id)
    session_id = temp_session_id()
    try:
        for attempt in range(1, attempts + 1):
            if reload_mode == "episode":
                session.policy_runner.refresh()
            elif reload_mode == "hot":
                session.policy_runner.refresh_if_due(interval_seconds=reload_interval_seconds)
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
                target_laps=target_laps,
                target_rank=target_rank,
                reload_during_episode=reload_mode == "hot",
                reload_interval_seconds=reload_interval_seconds,
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
                        target_laps=target_laps,
                        target_rank=target_rank,
                        reload_during_episode=reload_mode == "hot",
                        reload_interval_seconds=reload_interval_seconds,
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

        target_label = format_recording_target(
            target_laps=target_laps,
            target_rank=target_rank,
        )
        raise RuntimeError(f"No episode matched {target_label} after {attempts} attempts")
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
    target_laps: int,
    target_rank: int | None,
    reload_during_episode: bool,
    reload_interval_seconds: float,
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
        target_laps=target_laps,
        target_rank=target_rank,
    )
    progress.print(info, episode_return=episode_return, force=True)

    if video is None:
        while not (terminated or truncated):
            if reload_during_episode:
                policy_runner.refresh_if_due(interval_seconds=reload_interval_seconds)
            action = policy_runner.predict(
                observation,
                deterministic=deterministic,
                action_masks=env.action_masks(),
                refresh=False,
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
                if reload_during_episode:
                    policy_runner.refresh_if_due(interval_seconds=reload_interval_seconds)
                action = policy_runner.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                    refresh=False,
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
    race_laps_completed = int_info(info, "race_laps_completed")
    matched = _matches_recording_target(
        info,
        target_laps=target_laps,
        target_rank=target_rank,
    )
    return AttemptRunResult(
        attempt=attempt,
        path=path,
        matched=matched,
        finish_rank=finish_rank,
        race_laps_completed=race_laps_completed,
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
    target_laps: int,
    target_rank: int | None,
    reload_during_episode: bool,
    reload_interval_seconds: float,
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
        target_laps=target_laps,
        target_rank=target_rank,
        reload_during_episode=reload_during_episode,
        reload_interval_seconds=reload_interval_seconds,
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
        race_laps_completed=result.race_laps_completed,
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


def _resolve_recording_course_id(track_sampling: TrackSamplingConfig, requested: str) -> str:
    requested_key = _course_match_key(requested)
    if not requested_key:
        raise ValueError("--course-id must not be empty")
    if not track_sampling.enabled:
        raise ValueError("--course-id requires a managed run with track sampling enabled")

    available: list[str] = []
    for entry in track_sampling.entries:
        if not entry.course_id:
            continue
        if entry.course_id not in available:
            available.append(entry.course_id)
        aliases = (entry.course_id, entry.course_name, entry.display_name)
        if any(_course_match_key(alias) == requested_key for alias in aliases if alias):
            return entry.course_id

    if not available:
        raise ValueError("--course-id requires track sampling entries with course metadata")
    raise ValueError(
        f"course {requested!r} is not in this run's track pool. "
        f"Available course ids: {', '.join(available)}"
    )


def _course_match_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _finished_rank(info: dict[str, object]) -> int | None:
    if info.get("termination_reason") != "finished":
        return None
    rank = info.get("position")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        return None
    return rank


def _matches_recording_target(
    info: dict[str, object],
    *,
    target_laps: int,
    target_rank: int | None,
) -> bool:
    if int_info(info, "race_laps_completed") < target_laps:
        return False
    if target_rank is None:
        return True
    finish_rank = _finished_rank(info)
    return finish_rank is not None and finish_rank <= target_rank


def _print_attempt_result(result: RecordAttemptResult | AttemptRunResult) -> None:
    reason = result.termination_reason or result.truncation_reason or "unknown"
    rank = "-" if result.finish_rank is None else str(result.finish_rank)
    status = "keep" if result.matched else "discard"
    print(
        f"attempt {result.attempt}: {status}, reason={reason}, rank={rank}, "
        f"laps={result.race_laps_completed}, "
        f"time={format_race_time_ms(result.race_time_ms)}, "
        f"return={result.episode_return:.3f}, steps={result.episode_steps}"
    )
