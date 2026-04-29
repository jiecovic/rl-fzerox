# src/rl_fzerox/core/envs/info.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _MonitorInfoField:
    """One SB3 Monitor info field and its default value."""

    key: str
    default: object


_MONITOR_INFO_FIELDS: tuple[_MonitorInfoField, ...] = (
    _MonitorInfoField("episode_return", 0.0),
    _MonitorInfoField("episode_step", 0),
    _MonitorInfoField("termination_reason", None),
    _MonitorInfoField("truncation_reason", None),
    _MonitorInfoField("race_distance", None),
    _MonitorInfoField("speed_kph", None),
    _MonitorInfoField("position", None),
    _MonitorInfoField("total_racers", None),
    _MonitorInfoField("course_index", None),
    _MonitorInfoField("lap", None),
    _MonitorInfoField("laps_completed", None),
    _MonitorInfoField("race_laps_completed", None),
    _MonitorInfoField("raw_laps_completed", None),
    _MonitorInfoField("race_time_ms", None),
    _MonitorInfoField("damage_taken_frames", 0),
    _MonitorInfoField("collision_recoil_entered", False),
    _MonitorInfoField("boost_pad_entries", 0),
    _MonitorInfoField("boost_pad_entries_per_lap", None),
    _MonitorInfoField("episode_airborne_frames", 0),
    _MonitorInfoField("track_id", None),
    _MonitorInfoField("track_course_id", None),
    _MonitorInfoField("track_course_name", None),
)
MONITOR_INFO_KEYS: tuple[str, ...] = tuple(field.key for field in _MONITOR_INFO_FIELDS)


def ensure_monitor_info_keys(info: dict[str, object]) -> None:
    """Populate the SB3 Monitor info contract with stable defaults."""

    for field in _MONITOR_INFO_FIELDS:
        info.setdefault(field.key, field.default)
