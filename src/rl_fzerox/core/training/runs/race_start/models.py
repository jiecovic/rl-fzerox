# src/rl_fzerox/core/training/runs/race_start/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName

RaceStartMode: TypeAlias = Literal["time_attack", "gp_race"]


@dataclass(frozen=True, slots=True)
class RaceStartDefaults:
    """Race-start defaults used when the caller does not override a variant field."""

    lap_count: int = 3
    max_init_frames: int = 720


@dataclass(frozen=True, slots=True)
class RaceStartMenuTiming:
    """Frame timing used by the cold-boot menu materializer."""

    boot_frames: int = 300
    menu_ready_frames: int = 90
    post_unlock_settle_frames: int = 120
    start_hold_frames: int = 2
    start_settle_frames: int = 38
    menu_hold_frames: int = 8
    menu_settle_frames: int = 60
    mode_press_limit: int = 24
    race_init_frame_limit: int = 1_500


RACE_DEFAULTS = RaceStartDefaults()
MENU_TIMING = RaceStartMenuTiming()


@dataclass(frozen=True, slots=True)
class RaceStartVariant:
    """Target race-start setup materialized from an existing baseline state."""

    course_index: int
    mode: RaceStartMode
    character_index: int
    engine_setting_raw_value: int
    race_intro_target_timer: int | None
    gp_difficulty: RaceDifficultyName | None = None
    machine_select_slot: int | None = None
    total_lap_count: int = RACE_DEFAULTS.lap_count
