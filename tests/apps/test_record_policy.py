# tests/apps/test_record_policy.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any

import numpy as np
import pytest

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.apps.recording.cli import parse_args
from rl_fzerox.apps.recording.models import AttemptRunResult
from rl_fzerox.apps.recording.progress import (
    format_progress_line as _format_progress_line,
)
from rl_fzerox.apps.recording.progress import (
    format_race_time_ms as _format_race_time_ms,
)
from rl_fzerox.apps.recording.progress import (
    format_recording_target as _format_recording_target,
)
from rl_fzerox.apps.recording.runner import (
    _attempt_seed,
    _finished_rank,
    _matches_recording_target,
    _move_result_to_output,
    _resolve_recording_course_id,
    _run_attempt,
)
from rl_fzerox.apps.recording.video import (
    VideoSettings,
    _ffmpeg_command,
    as_pcm16_samples,
)
from rl_fzerox.apps.recording.video import (
    attempt_output_path as _attempt_output_path,
)
from rl_fzerox.apps.recording.video import (
    ensure_attempt_path_available as _ensure_attempt_path_available,
)
from rl_fzerox.apps.recording.video import (
    resolve_ffmpeg_path as _resolve_ffmpeg_path,
)
from rl_fzerox.apps.recording.video import (
    resolve_video_fps as _resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig


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


def test_resolve_video_fps_defaults_to_native_fps() -> None:
    assert _resolve_video_fps(native_fps=60.0, override=None) == 60.0


def test_resolve_video_fps_prefers_explicit_override() -> None:
    assert _resolve_video_fps(native_fps=60.0, override=120.0) == 120.0


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


def test_ffmpeg_command_streams_raw_rgb_to_low_latency_h264() -> None:
    command = _ffmpeg_command(
        ffmpeg_path="ffmpeg",
        output_path=Path(".race.attempt-007.mp4"),
        width=320,
        height=240,
        fps=60.0,
    )

    assert "-i" in command
    assert command[command.index("-i") + 1] == "-"
    assert "-tune" in command
    assert command[command.index("-tune") + 1] == "zerolatency"
    assert command[-1] == ".race.attempt-007.mp4"


def test_ffmpeg_command_streams_live_matroska_with_audio_pipe() -> None:
    command = _ffmpeg_command(
        ffmpeg_path="ffmpeg",
        output_path=Path("race.mkv"),
        width=320,
        height=240,
        fps=60.0,
        audio_sample_rate=48_000,
        audio_pipe_fd=7,
    )

    assert "pipe:7" in command
    assert "-ar" in command
    assert command[command.index("-ar") + 1] == "48000"
    assert "-c:a" in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-live" in command
    assert command[command.index("-live") + 1] == "1"
    assert "-cluster_time_limit" in command
    assert command[-1] == "race.mkv"


def test_as_pcm16_samples_requires_flat_stereo_pairs() -> None:
    samples = as_pcm16_samples((1, -1, 2, -2))

    assert samples.dtype == np.int16
    assert samples.tolist() == [1, -1, 2, -2]
    with pytest.raises(ValueError, match="even number"):
        as_pcm16_samples((1, 2, 3))
    with pytest.raises(ValueError, match="flat PCM"):
        as_pcm16_samples(np.zeros((1, 2), dtype=np.int16))


def test_resolve_ffmpeg_path_prefers_system_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rl_fzerox.apps.recording.video.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("rl_fzerox.apps.recording.video._imageio_ffmpeg_path", lambda: "/bundled")

    assert _resolve_ffmpeg_path() == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_path_uses_bundled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rl_fzerox.apps.recording.video.shutil.which", lambda _: None)
    monkeypatch.setattr("rl_fzerox.apps.recording.video._imageio_ffmpeg_path", lambda: "/bundled")

    assert _resolve_ffmpeg_path() == "/bundled"


def test_parse_args_records_deterministically_by_default() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4"])

    assert args.deterministic is True


def test_parse_args_can_record_stochastically() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4", "--no-deterministic"])

    assert args.deterministic is False


def test_parse_args_uses_live_progress_by_default() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4"])

    assert args.progress_interval == 2.0


def test_parse_args_can_disable_live_progress() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4", "--progress-interval", "0"])

    assert args.progress_interval == 0.0


def test_parse_args_defaults_to_stream_all_record_mode() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4"])

    assert args.record_mode == "stream-all"


def test_parse_args_accepts_probe_then_record_mode() -> None:
    args = parse_args(
        ["--run-dir", "run-dir", "--out", "race.mp4", "--record-mode", "probe-then-record"]
    )

    assert args.record_mode == "probe-then-record"


def test_parse_args_uses_lap_target_by_default() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4"])

    assert args.target_laps == 3
    assert args.target_rank is None


def test_parse_args_accepts_max_episode_alias_and_reload_mode() -> None:
    args = parse_args(
        [
            "--run-dir",
            "run-dir",
            "--out",
            "race.mp4",
            "--max-episodes",
            "12",
            "--reload-mode",
            "episode",
            "--reload-interval",
            "5",
        ]
    )

    assert args.max_episodes == 12
    assert args.reload_mode == "episode"
    assert args.reload_interval == 5.0


def test_parse_args_accepts_recording_course_id() -> None:
    args = parse_args(["--run-dir", "run-dir", "--out", "race.mp4", "--course-id", "space_plant"])

    assert args.course_id == "space_plant"


def test_resolve_recording_course_id_accepts_id_or_display_name() -> None:
    track_sampling = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="space_plant_time_attack_blue_falcon",
                course_id="space_plant",
                course_name="Space Plant",
                display_name="Space Plant Time Attack - Blue Falcon",
            ),
        ),
    )

    assert _resolve_recording_course_id(track_sampling, "space_plant") == "space_plant"
    assert _resolve_recording_course_id(track_sampling, "Space Plant") == "space_plant"


def test_resolve_recording_course_id_rejects_missing_course() -> None:
    track_sampling = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city_time_attack_blue_falcon",
                course_id="mute_city",
                course_name="Mute City",
            ),
        ),
    )

    with pytest.raises(ValueError, match="Available course ids: mute_city"):
        _resolve_recording_course_id(track_sampling, "space_plant")


def test_format_progress_line_shows_episode_state() -> None:
    line = _format_progress_line(
        {
            "episode_step": 123,
            "position": 4,
            "lap": 2,
            "race_time_ms": 83_456,
            "speed_kph": 987.6,
            "race_distance": 54321.0,
        },
        attempt=3,
        target_laps=3,
        target_rank=1,
        episode_return=42.125,
        effective_fps=123.4,
    )

    assert line == (
        "try 03 | step 123 | rank 4 | lap 2 | time 1:23.456 | "
        "988 km/h | 54.3k prog | 123.4 frames/s | R 42.1 | "
        "need laps >= 3, rank 1"
    )


def test_format_race_time_ms_handles_missing_and_regular_times() -> None:
    assert _format_race_time_ms(0) == "--:--.---"
    assert _format_race_time_ms(83_456) == "1:23.456"
    assert _format_race_time_ms(4_001) == "0:04.001"


def test_format_recording_target_shows_laps_and_optional_rank() -> None:
    assert _format_recording_target(target_laps=3, target_rank=None) == "need laps >= 3"
    assert _format_recording_target(target_laps=3, target_rank=1) == "need laps >= 3, rank 1"
    assert _format_recording_target(target_laps=3, target_rank=3) == "need laps >= 3, rank <= 3"


def test_recording_target_matches_lap_completion_without_rank() -> None:
    assert _matches_recording_target(
        {"race_laps_completed": 1, "termination_reason": "crashed", "position": 30},
        target_laps=1,
        target_rank=None,
    )


def test_recording_target_applies_optional_rank_filter() -> None:
    assert _matches_recording_target(
        {"race_laps_completed": 3, "termination_reason": "finished", "position": 2},
        target_laps=3,
        target_rank=3,
    )
    assert not _matches_recording_target(
        {"race_laps_completed": 3, "termination_reason": "finished", "position": 4},
        target_laps=3,
        target_rank=3,
    )


def test_attempt_seed_is_stable_per_attempt() -> None:
    assert _attempt_seed(100, 1) == 100
    assert _attempt_seed(100, 3) == 102
    assert _attempt_seed(None, 3) is None


def test_move_result_to_output_keeps_existing_file_if_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.mp4"
    output_path = tmp_path / "race.mp4"
    attempt_path.write_bytes(b"attempt")
    output_path.write_bytes(b"previous")

    def fail_replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        _move_result_to_output(
            AttemptRunResult(
                attempt=1,
                path=attempt_path,
                matched=True,
                finish_rank=1,
                race_laps_completed=3,
                episode_return=123.0,
                episode_steps=456,
                race_time_ms=78_900,
                termination_reason="finished",
                truncation_reason=None,
            ),
            output_path,
        )

    assert output_path.read_bytes() == b"previous"


def test_run_attempt_steps_policy_action_not_decoded_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeWriter:
        def __init__(self, **_: object) -> None:
            self.frames_written = 0

        def __enter__(self) -> FakeWriter:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def write(self, frame: object) -> None:
            del frame
            self.frames_written += 1

    class FakePolicyRunner:
        checkpoint_curriculum_stage_index: int | None = None
        refresh_if_due_called = False

        def reset(self) -> None:
            return None

        def refresh_if_due(self, *, interval_seconds: float) -> None:
            assert interval_seconds == 10.0
            self.refresh_if_due_called = True

        def predict(
            self,
            observation: object,
            *,
            deterministic: bool,
            action_masks: object,
            refresh: bool,
        ) -> int:
            del observation, deterministic, action_masks
            assert refresh is False
            return 7

    class FakeEnv:
        def __init__(self) -> None:
            self.stepped_action: object | None = None
            self.step_control_called = False

        def reset(self, *, seed: int | None = None) -> tuple[object, dict[str, object]]:
            assert seed == 123
            return object(), {"episode_step": 0}

        def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
            assert stage_index is None

        def render(self) -> RgbFrame:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def action_masks(self) -> None:
            return None

        def step_watch(self, action: object) -> SimpleNamespace:
            self.stepped_action = action
            return SimpleNamespace(
                observation=object(),
                reward=1.5,
                terminated=True,
                truncated=False,
                info={
                    "termination_reason": "finished",
                    "position": 1,
                    "race_laps_completed": 3,
                    "episode_step": 1,
                    "race_time_ms": 1234,
                },
                display_frames=(self.render(),),
            )

        def step_control(
            self,
            control_state: object,
        ) -> tuple[object, float, bool, bool, dict[str, object]]:
            del control_state
            self.step_control_called = True
            raise AssertionError("recording must not bypass env.step(action)")

    monkeypatch.setattr("rl_fzerox.apps.recording.runner.FfmpegRgbWriter", FakeWriter)
    env: Any = FakeEnv()
    policy_runner: Any = FakePolicyRunner()

    result = _run_attempt(
        env,
        policy_runner=policy_runner,
        path=tmp_path / "attempt.mp4",
        attempt=1,
        seed=123,
        deterministic=True,
        target_laps=3,
        target_rank=1,
        reload_during_episode=True,
        reload_interval_seconds=10.0,
        video=VideoSettings(
            path=tmp_path / "attempt.mp4",
            ffmpeg_path="ffmpeg",
            fps=60.0,
        ),
        progress_interval_seconds=0.0,
    )

    assert env.stepped_action == 7
    assert env.step_control_called is False
    assert policy_runner.refresh_if_due_called is True
    assert result.matched is True
    assert result.race_laps_completed == 3
    assert result.episode_return == pytest.approx(1.5)
