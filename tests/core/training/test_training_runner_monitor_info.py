# tests/core/training/test_training_runner_monitor_info.py
from __future__ import annotations

from rl_fzerox.core.envs.info import MONITOR_INFO_KEYS


def test_monitor_info_keys_include_position_context() -> None:
    assert "position" in MONITOR_INFO_KEYS
    assert "ko_star_count" in MONITOR_INFO_KEYS
    assert "total_racers" in MONITOR_INFO_KEYS
    assert "course_index" in MONITOR_INFO_KEYS


def test_monitor_info_keys_include_finished_timing_and_damage_metrics() -> None:
    assert "race_time_ms" in MONITOR_INFO_KEYS
    assert "episode_completion_fraction" in MONITOR_INFO_KEYS
    assert "damage_taken_frames" in MONITOR_INFO_KEYS
    assert "boost_pad_entries" in MONITOR_INFO_KEYS
    assert "boost_pad_entries_per_lap" in MONITOR_INFO_KEYS
    assert "episode_airborne_frames" in MONITOR_INFO_KEYS


def test_monitor_info_keys_exclude_step_only_action_rates() -> None:
    assert "boost_used" not in MONITOR_INFO_KEYS
    assert "lean_used" not in MONITOR_INFO_KEYS
    assert "boost_pad_entered" not in MONITOR_INFO_KEYS


def test_monitor_info_keys_include_episode_action_masks() -> None:
    assert "lean_episode_masked" in MONITOR_INFO_KEYS


def test_monitor_info_keys_include_track_context_for_course_metrics() -> None:
    assert "track_id" in MONITOR_INFO_KEYS
    assert "track_course_id" in MONITOR_INFO_KEYS
    assert "track_course_name" in MONITOR_INFO_KEYS
