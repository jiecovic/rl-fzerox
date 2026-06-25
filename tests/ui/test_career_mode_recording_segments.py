# tests/ui/test_career_mode_recording_segments.py
"""Segment lifecycle tests for Career Mode recording.

The scenarios cover attempt boundaries, retries, terminal result frames,
partial close behavior, and summary status emitted for completed segments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_fzerox.ui.watch.runtime.career_mode.recording import CareerModeFrameRecorder
from tests.ui.career_mode_recording_support import _FakeFinalizer, _FakeWriter


def test_career_recorder_keeps_segment_during_post_result_continuation(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={"career_mode_target_label": "Clear Novice Jack Cup"},
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_target_label": "Clear Novice Queen Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_target_label": "Clear Novice Queen Cup",
            "career_mode_fsm_continuing_result": False,
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-queen-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-partial-clear-novice-queen-cup.mkv",
    ]


def test_career_recorder_finishes_failed_segment_after_terminal_result_frame(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": False,
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "crashed",
        },
    )

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [2, 2]
    assert writers[1].closed
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv",
    ]


def test_career_recorder_finishes_failed_segment_from_terminal_event(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "crashed",
        },
    )

    assert writers[1].closed
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv",
    ]


def test_career_recorder_discards_failed_segments_when_requested(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        keep_failed_segments=False,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )

    assert writers[1].closed
    assert not (tmp_path / "career.segment-001-clear-master-joker-cup.mkv").exists()
    assert finalizer.paths == []


def test_career_recorder_restarts_segment_after_next_attempt_retry(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-jack-cup.mkv",
        "career.segment-002-clear-master-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [4, 2, 2]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-master-jack-cup.mkv",
        "career.segment-002-partial-clear-master-jack-cup.mkv",
    ]


def test_career_recorder_keeps_course_retire_inside_current_segment(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_terminal_result": True,
            "career_mode_race_terminal": True,
            "termination_reason": "retired",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [4, 4]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-partial-clear-master-joker-cup.mkv",
    ]

    summary_json_path = tmp_path / "career.segment-001-partial-clear-master-joker-cup.summary.json"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] is None
    assert summary["result_counts"] == {"crashed": 0, "failed": 1, "finished": 0, "retired": 1}


def test_career_recorder_segments_by_attempt_and_marks_failed_attempts(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": False,
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [4, 2, 2]
    assert not (tmp_path / "career.segment-001-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-002-partial-clear-novice-jack-cup.mkv").exists()
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv",
        "career.segment-002-partial-clear-novice-jack-cup.mkv",
    ]


def test_career_recorder_starts_new_segment_after_succeeded_replay_attempt(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master King Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master King Cup",
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master King Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Master King Cup",
            "career_mode_fsm_continuing_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-king-cup.mkv",
        "career.segment-002-clear-master-king-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-king-cup.mkv",
        "career.segment-002-partial-clear-master-king-cup.mkv",
    ]


def test_career_recorder_finishes_succeeded_segment_from_progress_event(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "game_mode": "main_menu",
            "termination_reason": "finished",
        },
    )

    assert writers[1].closed
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]


def test_career_recorder_ignores_frames_from_closed_attempt(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "game_mode": "gp_end_cutscene",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
            "game_mode": "gp_end_cutscene",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]


def test_career_recorder_finishes_clean_cup_after_menu_exit_frame(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
            "termination_reason": "finished",
            "track_course_id": "big_hand",
            "track_course_name": "Big Hand",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "game_mode": "main_menu",
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "game_mode": "main_menu",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "game_mode": "main_menu",
        },
    )

    assert writers[1].closed
    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert finalizer.summaries[-1] is not None
    assert finalizer.summaries[-1].status == "succeeded"


def test_career_recorder_finishes_failed_cup_after_game_over_frame(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Expert King Cup",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Expert King Cup",
            "termination_reason": "retired",
            "track_course_id": "white_land",
            "track_course_name": "White Land",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Expert King Cup",
            "game_mode": "game_over",
        },
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Expert King Cup",
            "game_mode": "game_over",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Expert King Cup",
            "game_mode": "game_over",
        },
    )

    assert writers[1].closed
    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-expert-king-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-expert-king-cup.mkv",
    ]
    assert finalizer.summaries[-1] is not None
    assert finalizer.summaries[-1].status == "failed"


def test_career_recorder_finalizes_succeeded_segment_on_post_gp_completion(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_continuing_result": False,
            "career_mode_fsm_terminal_result": True,
            "career_mode_race_terminal": True,
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "race_time_ms": 92_345,
            "position": 1,
            "ko_star_count": 3,
            "track_course_id": "big_hand",
            "track_course_name": "Big Hand",
            "track_course_index": 23,
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 80,
        },
    )

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert not writers[1].closed
    assert finalizer.paths == []

    post_gp_info = {
        "career_mode_attempt_id": None,
        "career_mode_target_label": "Clear Master Joker Cup",
        "game_mode": "gp_end_cutscene",
        "career_mode_fsm_observed_screen": "post_gp",
        "career_mode_fsm_continuing_result": False,
        "career_mode_last_finished_attempt_id": "attempt-final",
        "career_mode_last_finished_attempt_status": "succeeded",
    }
    recorder.record_frame(
        frame,
        info={
            **post_gp_info,
            "career_mode_gp_final_rank": 1,
            "career_mode_gp_points": 512,
            "gp_final_rank": -15533,
            "gp_points": -32640,
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={**post_gp_info, "gp_final_rank": -15533, "gp_points": -32640},
    )

    assert writers[1].closed
    assert len(writers[1].frames) == 3
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert recorder.drain_notices() == ("MP4 ready: career.segment-001-clear-master-joker-cup.mp4",)
    summary_json_path = tmp_path / "career.segment-001-clear-master-joker-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-clear-master-joker-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["label"] == "Clear Master Joker Cup"
    assert summary["status"] == "succeeded"
    assert summary["video"]["frame_count"] == 3
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 1, "retired": 0}
    assert "-15533" not in summary_json_path.read_text(encoding="utf-8")
    assert "-32640" not in summary_json_path.read_text(encoding="utf-8")
    assert summary["summary"] == {
        "average_position": 1.0,
        "best_position": 1,
        "course_attempt_count": 1,
        "crashed_course_count": 0,
        "failed_attempt_count": 0,
        "final_gp_position": 1,
        "finished_attempt_count": 1,
        "gp_points": 512,
        "ko_star_total": 3,
        "retired_course_count": 0,
        "total_race_time_ms": 92_345,
        "unique_finished_course_count": 1,
        "worst_position": 1,
    }
    assert summary["courses"] == [
        {
            "course_id": "big_hand",
            "course_index": 23,
            "course_name": "Big Hand",
            "difficulty": "master",
            "engine_setting_raw_value": 80,
            "ko_star_count": 3,
            "position": 1,
            "race_time_ms": 92_345,
            "termination_reason": "finished",
            "track_id": "big_hand",
            "vehicle_name": "Blue Falcon",
        }
    ]
    assert "| Big Hand | finished | 1:32.345 | 1 | 3 | Engine 63 |" in (
        summary_md_path.read_text(encoding="utf-8")
    )
    assert "| Final GP position | 1 |" in summary_md_path.read_text(encoding="utf-8")
    assert "| GP points | 512 |" in summary_md_path.read_text(encoding="utf-8")
    assert "-15533" not in summary_md_path.read_text(encoding="utf-8")
    assert "-32640" not in summary_md_path.read_text(encoding="utf-8")

    recorder.close()

    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert recorder.drain_notices() == ()
