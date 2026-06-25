# tests/ui/test_career_mode_recording_paths.py
"""Path naming tests for Career Mode recording artifacts.

The cases lock down label sanitization, manager attempt identifiers, and the
partial-close marker used when recording windows end before success/failure.
"""

from pathlib import Path

from rl_fzerox.ui.watch.runtime.career_mode.recording import career_segment_recording_path


def test_career_segment_recording_path_sanitizes_label() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
        ).name
        == "career.segment-007-clear-expert-joker-cup.mkv"
    )


def test_career_segment_recording_path_includes_manager_attempt_uid() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
            attempt_id="20260621-unlock-save-attempt-054792f5",
        ).name
        == "career.segment-007-054792f5-clear-expert-joker-cup.mkv"
    )


def test_career_segment_recording_path_marks_partial_window_close() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
            status="failed",
            partial=True,
        ).name
        == "career.segment-007-failed-attempt-partial-clear-expert-joker-cup.mkv"
    )
