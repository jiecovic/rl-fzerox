# src/rl_fzerox/core/envs/info.py
from __future__ import annotations

MONITOR_INFO_KEYS: tuple[str, ...] = (
    "episode_return",
    "episode_step",
    "termination_reason",
    "truncation_reason",
    "race_distance",
    "speed_kph",
    "position",
    "lap",
    "laps_completed",
    "race_laps_completed",
    "raw_laps_completed",
    "race_time_ms",
    "damage_taken_frames",
    "collision_recoil_entered",
    "milestones_completed",
)


def ensure_monitor_info_keys(info: dict[str, object]) -> None:
    """Populate the SB3 Monitor info contract with stable defaults."""

    info.setdefault("episode_return", 0.0)
    info.setdefault("episode_step", 0)
    info.setdefault("termination_reason", None)
    info.setdefault("truncation_reason", None)
    info.setdefault("race_distance", None)
    info.setdefault("speed_kph", None)
    info.setdefault("position", None)
    info.setdefault("lap", None)
    info.setdefault("laps_completed", None)
    info.setdefault("race_laps_completed", None)
    info.setdefault("raw_laps_completed", None)
    info.setdefault("race_time_ms", None)
    info.setdefault("damage_taken_frames", 0)
    info.setdefault("collision_recoil_entered", False)
    info.setdefault("milestones_completed", None)
