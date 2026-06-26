# tests/ui/test_career_mode_recording_summaries.py
"""Summary payload regression tests for Career Mode recording.

These tests focus on terminal event aggregation, retry bookkeeping, duplicate
terminal samples, and stale terminal frame filtering in generated summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_fzerox.ui.watch.runtime.career_mode.recording import CareerModeFrameRecorder
from tests.ui.career_mode_recording_support import _FakeFinalizer, _FakeWriter


def test_career_recorder_summary_captures_terminal_events_between_frames(
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
            "career_mode_policy_artifact": "latest",
            "career_mode_policy_checkpoint_local_num_timesteps": 660_000,
            "career_mode_policy_checkpoint_mtime_ns": 1_765_275_200_000_000_000,
            "career_mode_policy_checkpoint_mtime_utc": "2025-12-10T12:00:00Z",
            "career_mode_policy_checkpoint_num_timesteps": 14_820_470,
            "career_mode_policy_checkpoint_path": (
                str(tmp_path / "runs" / "run-a" / "checkpoints" / "latest" / "policy.zip")
            ),
            "career_mode_policy_course_id": "mute_city",
            "career_mode_policy_source_id": "run-a",
            "career_mode_policy_source_kind": "run",
            "career_mode_policy_source_name": "Run A",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "termination_reason": "finished",
            "race_time_ms": 81_234,
            "position": 1,
            "ko_star_count": 2,
            "track_course_id": "mute_city",
            "track_course_name": "Mute City",
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 50,
        }
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "race_time_ms": 85_678,
            "position": 2,
            "ko_star_count": 4,
            "track_course_id": "silence",
            "track_course_name": "Silence",
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 60,
        }
    )
    recorder.close()

    summary_json_path = tmp_path / "career.segment-001-partial-clear-master-jack-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-partial-clear-master-jack-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["video"]["frame_count"] == 2
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 2, "retired": 0}
    assert summary["policy_checkpoints"] == [
        {
            "artifact": "latest",
            "course_id": "mute_city",
            "local_num_timesteps": 660_000,
            "mtime_ns": 1_765_275_200_000_000_000,
            "mtime_utc": "2025-12-10T12:00:00Z",
            "num_timesteps": 14_820_470,
            "path": str(tmp_path / "runs" / "run-a" / "checkpoints" / "latest" / "policy.zip"),
            "source_id": "run-a",
            "source_kind": "run",
            "source_name": "Run A",
        }
    ]
    assert summary["courses"] == [
        {
            "course_id": "mute_city",
            "course_name": "Mute City",
            "difficulty": "master",
            "engine_setting_raw_value": 50,
            "ko_star_count": 2,
            "position": 1,
            "race_time_ms": 81_234,
            "termination_reason": "finished",
            "vehicle_name": "Blue Falcon",
        },
        {
            "course_id": "silence",
            "course_name": "Silence",
            "difficulty": "master",
            "engine_setting_raw_value": 60,
            "ko_star_count": 4,
            "position": 2,
            "race_time_ms": 85_678,
            "termination_reason": "finished",
            "vehicle_name": "Blue Falcon",
        },
    ]
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "| Mute City | finished | 1:21.234 | 1 | 2 | Engine 39 |" in markdown
    assert "| Silence | finished | 1:25.678 | 2 | 4 | Engine 47 |" in markdown
    assert "| Run A | latest | mute_city | 14820470 | 660000 | 2025-12-10T12:00:00Z |" in markdown


def test_career_recorder_summary_keeps_retried_course_results(
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

    base_info = {
        "career_mode_attempt_id": "attempt-a",
        "career_mode_target_label": "Clear Master Jack Cup",
        "course_index": 0,
        "track_engine_setting_raw_value": 80,
    }
    recorder.record_frame(frame, info=base_info)
    recorder.record_event(
        info={
            **base_info,
            "termination_reason": "retired",
            "race_time_ms": 41_250,
            "position": 30,
            "ko_star_count": 0,
            "race_laps_completed": 1,
            "total_lap_count": 3,
        }
    )
    recorder.record_event(
        info={
            **base_info,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "race_time_ms": 86_136,
            "position": 1,
            "ko_star_count": 1,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            **base_info,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "career_mode_gp_final_rank": 1,
            "career_mode_gp_points": 600,
            "game_mode": "gp_end_cutscene",
        },
    )
    recorder.close()

    summary_json_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["result_counts"] == {"crashed": 0, "failed": 1, "finished": 1, "retired": 1}
    assert [course["termination_reason"] for course in summary["courses"]] == [
        "retired",
        "finished",
    ]
    assert [course["course_name"] for course in summary["courses"]] == ["Mute City", "Mute City"]
    assert summary["summary"]["final_gp_position"] == 1
    assert summary["summary"]["gp_points"] == 600
    assert summary["summary"]["course_attempt_count"] == 2
    assert summary["summary"]["finished_attempt_count"] == 1
    assert summary["summary"]["failed_attempt_count"] == 1
    assert summary["summary"]["unique_finished_course_count"] == 1
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "| Course attempts | 2 |" in markdown
    assert "| Failed attempts | 1 |" in markdown
    assert "| Mute City | retired | 0:41.250 | 30 | 0 | Engine 63 |" in markdown
    assert "| Mute City | finished | 1:26.136 | 1 | 1 | Engine 63 |" in markdown


def test_career_recorder_summary_deduplicates_duplicate_terminal_observations(
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
    for course_index, race_time_ms, position, ko_stars, engine_raw in (
        (0, 86_136, 7, 0, 103),
        (1, 78_655, 11, 0, 103),
        (2, 70_828, 1, 0, 103),
        (3, 82_910, 4, 0, 103),
        (4, 86_007, 1, 0, 103),
    ):
        recorder.record_event(
            info={
                "career_mode_attempt_id": "attempt-a",
                "career_mode_target_label": "Clear Master Jack Cup",
                "game_mode": "gp_race",
                "course_index": course_index,
                "engine_setting_raw_value_ram": engine_raw,
            }
        )
        terminal_info = {
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "termination_reason": "finished",
            "race_time_ms": race_time_ms,
            "position": position,
            "ko_star_count": ko_stars,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
        recorder.record_event(info=terminal_info)
        recorder.record_event(
            info={
                **terminal_info,
                "course_index": course_index,
                "engine_setting_raw_value_ram": 50,
            }
        )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "game_mode": "gp_race",
            "course_index": 5,
            "engine_setting_raw_value_ram": 103,
        }
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "finished",
            "race_time_ms": 85_673,
            "position": 1,
            "ko_star_count": 1,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "finished",
            "race_time_ms": 85_673,
            "position": 1,
            "ko_star_count": 1,
            "race_laps_completed": 3,
            "total_lap_count": 3,
            "course_index": 5,
            "engine_setting_raw_value_ram": 50,
        }
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "finished",
            "race_time_ms": 130_340,
            "position": 1,
            "ko_star_count": 0,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        }
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "termination_reason": "finished",
            "race_time_ms": 131_000,
            "position": 1,
            "ko_star_count": 3,
            "race_laps_completed": 3,
            "total_lap_count": 3,
            "course_index": 55,
        }
    )
    recorder.finish_segment(
        status="failed",
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
            "game_mode": "main_menu",
        },
    )
    recorder.close()

    summary_json_path = (
        tmp_path / "career.segment-001-failed-attempt-clear-master-jack-cup.summary.json"
    )
    summary_md_path = (
        tmp_path / "career.segment-001-failed-attempt-clear-master-jack-cup.summary.md"
    )
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 6, "retired": 0}
    assert summary["summary"]["course_attempt_count"] == 6
    assert summary["summary"]["total_race_time_ms"] == 490_209
    assert summary["summary"]["average_position"] == 25 / 6
    assert summary["summary"]["gp_points"] is None
    assert summary["summary"]["ko_star_total"] == 1
    assert [course["course_name"] for course in summary["courses"]] == [
        "Mute City",
        "Silence",
        "Sand Ocean",
        "Devil's Forest",
        "Big Blue",
        "Port Town",
    ]
    assert [course["position"] for course in summary["courses"]] == [7, 11, 1, 4, 1, 1]
    assert {course["engine_setting_raw_value"] for course in summary["courses"]} == {103}
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "| - | finished |" not in markdown
    assert "| Mute City | finished | 1:26.136 | 7 | 0 | Engine 80 |" in markdown


def test_career_recorder_summary_ignores_stale_terminal_frame_samples(
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

    base_info = {
        "career_mode_attempt_id": "attempt-a",
        "career_mode_target_label": "Clear Master Jack Cup",
        "game_mode": "gp_race",
    }
    recorder.record_frame(
        frame,
        info={
            **base_info,
            "course_index": 0,
            "engine_setting_raw_value_ram": 80,
        },
    )
    recorder.record_event(
        info={
            **base_info,
            "termination_reason": "finished",
            "course_index": 0,
            "race_time_ms": 76_120,
            "position": 1,
            "ko_star_count": 0,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )
    recorder.record_frame(
        frame,
        info={
            **base_info,
            "termination_reason": "finished",
            "course_index": 1,
            "race_time_ms": 76_120,
            "position": 30,
            "ko_star_count": 0,
            "race_laps_completed": 3,
            "total_lap_count": 3,
            "engine_setting_raw_value_ram": 50,
        },
    )
    recorder.record_frame(
        frame,
        info={
            **base_info,
            "course_index": 1,
            "engine_setting_raw_value_ram": 96,
        },
    )
    recorder.record_event(
        info={
            **base_info,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "course_index": 1,
            "race_time_ms": 68_233,
            "position": 1,
            "ko_star_count": 0,
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
    )
    recorder.finish_segment(
        status="succeeded",
        info={
            **base_info,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "game_mode": "main_menu",
        },
    )
    recorder.close()

    summary_json_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 2, "retired": 0}
    assert [course["course_name"] for course in summary["courses"]] == ["Mute City", "Silence"]
    assert [course["race_time_ms"] for course in summary["courses"]] == [76_120, 68_233]
    assert [course["position"] for course in summary["courses"]] == [1, 1]
    assert [course["engine_setting_raw_value"] for course in summary["courses"]] == [80, 96]
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "| Mute City | finished | 1:16.120 | 1 | 0 | Engine 63 |" in markdown
    assert "| Silence | finished | 1:08.233 | 1 | 0 | Engine 75 |" in markdown
    assert "| Silence | finished | 1:16.120 | 30 | 0 |" not in markdown
