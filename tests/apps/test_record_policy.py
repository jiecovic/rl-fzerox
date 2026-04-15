# tests/apps/test_record_policy.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.record_policy import (
    _attempt_output_path,
    _attempt_seed,
    _ensure_attempt_path_available,
    _ffmpeg_command,
    _finished_rank,
    _format_progress_line,
    _format_target_rank,
    _resolve_ffmpeg_path,
    _resolve_video_fps,
    parse_args,
)


def test_finished_rank_accepts_finished_positive_rank() -> None:
    assert _finished_rank({"termination_reason": "finished", "position": 1}) == 1


@pytest.mark.parametrize(
    "info",
    [
        {"termination_reason": "progress_stalled", "position": 1},
        {"termination_reason": "finished", "position": 0},
        {"termination_reason": "finished", "position": True},
        {"termination_reason": "finished", "position": "1"},
    ],
)
def test_finished_rank_rejects_non_matching_episode(info: dict[str, object]) -> None:
    assert _finished_rank(info) is None


def test_resolve_video_fps_defaults_to_native_fps_per_action_repeat() -> None:
    assert _resolve_video_fps(native_fps=60.0, action_repeat=2, override=None) == 30.0


def test_resolve_video_fps_prefers_explicit_override() -> None:
    assert _resolve_video_fps(native_fps=60.0, action_repeat=2, override=120.0) == 120.0


def test_attempt_output_path_keeps_attempts_next_to_final_output() -> None:
    path = _attempt_output_path(Path("/tmp/out/race.mp4"), 7)

    assert path == Path("/tmp/out/.race.attempt-007.mp4")


def test_attempt_output_path_can_include_session_id() -> None:
    path = _attempt_output_path(
        Path("/tmp/out/race.mp4"),
        7,
        session_id="session-123-456",
    )

    assert path == Path("/tmp/out/.race.session-123-456.attempt-007.mp4")


def test_ensure_attempt_path_available_rejects_existing_temp_mp4(tmp_path: Path) -> None:
    attempt_path = tmp_path / ".race.attempt-007.mp4"
    attempt_path.write_bytes(b"unfinished")

    with pytest.raises(FileExistsError):
        _ensure_attempt_path_available(attempt_path)


def test_ffmpeg_command_streams_raw_rgb_to_h264_mp4() -> None:
    command = _ffmpeg_command(
        ffmpeg_path="ffmpeg",
        output_path=Path(".race.attempt-007.mp4"),
        width=320,
        height=240,
        fps=60.0,
    )

    assert "-i" in command
    assert command[command.index("-i") + 1] == "-"
    assert command[-1] == ".race.attempt-007.mp4"


def test_resolve_ffmpeg_path_prefers_system_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rl_fzerox.apps.record_policy.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("rl_fzerox.apps.record_policy._imageio_ffmpeg_path", lambda: "/bundled")

    assert _resolve_ffmpeg_path() == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_path_uses_bundled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rl_fzerox.apps.record_policy.shutil.which", lambda _: None)
    monkeypatch.setattr("rl_fzerox.apps.record_policy._imageio_ffmpeg_path", lambda: "/bundled")

    assert _resolve_ffmpeg_path() == "/bundled"


def test_parse_args_records_deterministically_by_default() -> None:
    args = parse_args(["--out", "race.mp4"])

    assert args.deterministic is True


def test_parse_args_can_record_stochastically() -> None:
    args = parse_args(["--out", "race.mp4", "--no-deterministic"])

    assert args.deterministic is False


def test_parse_args_uses_live_progress_by_default() -> None:
    args = parse_args(["--out", "race.mp4"])

    assert args.progress_interval == 2.0


def test_parse_args_can_disable_live_progress() -> None:
    args = parse_args(["--out", "race.mp4", "--progress-interval", "0"])

    assert args.progress_interval == 0.0


def test_format_progress_line_shows_episode_state() -> None:
    line = _format_progress_line(
        {
            "episode_step": 123,
            "position": 4,
            "lap": 2,
            "speed_kph": 987.6,
            "race_distance": 54321.0,
        },
        attempt=3,
        target_rank=1,
        episode_return=42.125,
        effective_fps=123.4,
    )

    assert line == (
        "try 03 | step 123 | rank 4 | lap 2 | 988 km/h | 54.3k prog | "
        "123.4 frames/s | R 42.1 | need rank 1"
    )


def test_format_target_rank_special_cases_first_place() -> None:
    assert _format_target_rank(1) == "need rank 1"
    assert _format_target_rank(3) == "need rank <= 3"


def test_attempt_seed_is_stable_per_attempt() -> None:
    assert _attempt_seed(100, 1) == 100
    assert _attempt_seed(100, 3) == 102
    assert _attempt_seed(None, 3) is None
