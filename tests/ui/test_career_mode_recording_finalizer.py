# tests/ui/test_career_mode_recording_finalizer.py
"""MP4 finalizer tests for Career Mode recording artifacts.

These tests isolate remuxing, session-summary generation, optional live MP4
creation, and warning behavior when ffmpeg output is missing or fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    _Mp4RecordingFinalizer,
    _SegmentSummarySnapshot,
    career_session_summary_path,
)


def test_career_mp4_finalizer_remuxes_mkv_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        calls.append((path, ffmpeg_path))
        output_path = path.with_suffix(".mp4")
        output_path.write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    finalizer = _Mp4RecordingFinalizer()
    mkv_path = tmp_path / "segment.mkv"

    finalizer.finalize(mkv_path)
    finalizer.finalize(tmp_path / "segment.mp4")
    finalizer.close()

    assert calls == [(mkv_path, "ffmpeg")]


def test_career_mp4_finalizer_writes_session_summary_for_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    concat_calls: list[tuple[tuple[Path, ...], str, Path]] = []

    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del ffmpeg_path
        output_path = path.with_suffix(".mp4")
        output_path.write_bytes(b"mp4")
        return output_path

    def fake_concat_mp4_recordings(
        input_paths: tuple[Path, ...],
        *,
        ffmpeg_path: str,
        output_path: Path,
    ) -> Path:
        concat_calls.append((tuple(input_paths), ffmpeg_path, output_path))
        output_path.write_bytes(b"session mp4")
        return output_path

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.concat_mp4_recordings",
        fake_concat_mp4_recordings,
    )
    session_path = tmp_path / "career.mkv"
    live_path = tmp_path / "career.live.mkv"
    segment_a = tmp_path / "career.segment-001-clear-master-jack-cup.mkv"
    segment_b = tmp_path / "career.segment-002-clear-master-jack-cup.mkv"
    finalizer = _Mp4RecordingFinalizer(
        session_source_path=session_path,
        live_source_path=live_path,
    )

    finalizer.finalize(
        segment_b,
        summary=_SegmentSummarySnapshot(
            segment_index=2,
            label="Clear Master Jack Cup",
            attempt_id="attempt-b",
            status="failed",
            source_path=segment_b,
            started_at_utc="2026-06-15T10:02:00Z",
            closed_at_utc="2026-06-15T10:03:00Z",
            frame_count=20,
            course_results=(
                {
                    "course_name": "Silence",
                    "termination_reason": "crashed",
                    "race_time_ms": 12_000,
                },
            ),
            final_info={},
        ),
    )
    finalizer.finalize(
        segment_a,
        summary=_SegmentSummarySnapshot(
            segment_index=1,
            label="Clear Master Jack Cup",
            attempt_id="attempt-a",
            status="succeeded",
            source_path=segment_a,
            started_at_utc="2026-06-15T10:00:00Z",
            closed_at_utc="2026-06-15T10:01:00Z",
            frame_count=10,
            course_results=(
                {
                    "course_name": "Mute City",
                    "termination_reason": "finished",
                    "race_time_ms": 81_234,
                },
            ),
            final_info={},
        ),
    )
    finalizer.close()

    payload = json.loads(career_session_summary_path(session_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "career_recording_session_summary"
    assert payload["session_source_path"] == str(session_path)
    assert payload["live_mkv_path"] == str(live_path)
    assert payload["session_mp4_path"] == str(tmp_path / "career.session.mp4")
    assert payload["segment_count"] == 2
    assert payload["result_counts"] == {"crashed": 1, "failed": 1, "finished": 1, "retired": 0}
    assert [segment["segment_index"] for segment in payload["segments"]] == [1, 2]
    assert [segment["video"]["mp4_path"] for segment in payload["segments"]] == [
        str(segment_a.with_suffix(".mp4")),
        str(segment_b.with_suffix(".mp4")),
    ]
    assert concat_calls == [
        (
            (segment_a.with_suffix(".mp4"), segment_b.with_suffix(".mp4")),
            "ffmpeg",
            tmp_path / "career.session.mp4",
        )
    ]


def test_career_mp4_finalizer_can_skip_session_mp4_for_single_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del ffmpeg_path
        output_path = path.with_suffix(".mp4")
        output_path.write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    session_path = tmp_path / "career.mkv"
    segment_path = tmp_path / "career.segment-001-clear-master-jack-cup.mkv"
    finalizer = _Mp4RecordingFinalizer(
        session_source_path=session_path,
        session_mp4_enabled=False,
    )

    finalizer.finalize(
        segment_path,
        summary=_SegmentSummarySnapshot(
            segment_index=1,
            label="Clear Master Jack Cup",
            attempt_id="attempt-a",
            status="succeeded",
            source_path=segment_path,
            started_at_utc="2026-06-15T10:00:00Z",
            closed_at_utc="2026-06-15T10:01:00Z",
            frame_count=10,
            course_results=(),
            final_info={},
        ),
    )
    finalizer.close()

    payload = json.loads(career_session_summary_path(session_path).read_text(encoding="utf-8"))
    assert payload["session_mp4_path"] is None
    assert payload["live_mkv_path"] is None
    assert payload["segment_count"] == 1


def test_career_mp4_finalizer_reports_close_time_remux_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del path, ffmpeg_path
        raise RuntimeError("remux failed")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    finalizer = _Mp4RecordingFinalizer()

    finalizer.finalize(tmp_path / "segment.mkv")
    finalizer.close()

    assert finalizer.drain_notices() == ("MP4 conversion failed: remux failed",)


def test_career_mp4_finalizer_keeps_mkv_when_remux_output_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del ffmpeg_path
        return path.with_suffix(".mp4")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    session_path = tmp_path / "career.mkv"
    segment_path = tmp_path / "career.segment-001-clear-master-joker-cup.mkv"
    segment_path.write_bytes(b"mkv")
    finalizer = _Mp4RecordingFinalizer(session_source_path=session_path)

    finalizer.finalize(
        segment_path,
        summary=_SegmentSummarySnapshot(
            segment_index=1,
            label="Clear Master Joker Cup",
            attempt_id="attempt-a",
            status="succeeded",
            source_path=segment_path,
            started_at_utc="2026-06-17T10:00:00Z",
            closed_at_utc="2026-06-17T10:01:00Z",
            frame_count=10,
            course_results=(),
            final_info={},
        ),
    )
    finalizer.close()

    notices = finalizer.drain_notices()
    payload = json.loads(career_session_summary_path(session_path).read_text(encoding="utf-8"))
    assert segment_path.exists()
    assert notices == (
        f"MP4 conversion failed: MP4 output was not created: {segment_path.with_suffix('.mp4')}",
    )
    assert payload["segment_count"] == 0
