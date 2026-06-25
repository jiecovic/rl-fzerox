# src/rl_fzerox/core/career_mode/navigation/types.py
"""Shared Career Mode navigation enums and immutable timing values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MenuInput(StrEnum):
    """Semantic menu inputs before mapping them to emulator controller buttons."""

    NEUTRAL = "neutral"
    ACCEPT = "accept"
    A_BUTTON = "a_button"
    CANCEL = "cancel"
    START = "start"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NEXT_CAMERA = "next_camera"


class CareerPhase(StrEnum):
    """High-level control state for one Career Mode race attempt."""

    BOOT_TO_DIFFICULTY = "boot_to_difficulty"
    SELECT_DIFFICULTY = "select_difficulty"
    ENTER_COURSE_SELECT = "enter_course_select"
    SELECT_CUP = "select_cup"
    ENTER_MACHINE_SELECT = "enter_machine_select"
    SELECT_MACHINE = "select_machine"
    ENTER_MACHINE_SETTINGS = "enter_machine_settings"
    APPLY_ENGINE = "apply_engine"
    ENTER_RACE = "enter_race"
    POLICY_RACE = "policy_race"
    CONTINUE_AFTER_RACE = "continue_after_race"


class DifficultyPopupState(StrEnum):
    """Controller-owned state for the GP difficulty overlay."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    SUBMITTED = "submitted"


class ObservedMenuScreen(StrEnum):
    """Visible game screen inferred from native telemetry."""

    TITLE = "title"
    MAIN_MENU_GP = "main_menu_gp"
    MAIN_MENU_OTHER = "main_menu_other"
    DIFFICULTY_POPUP = "difficulty_popup"
    DIFFICULTY_CONFIRM = "difficulty_confirm"
    TRANSITION = "transition"
    COURSE_SELECT = "course_select"
    MACHINE_SELECT = "machine_select"
    MACHINE_SETTINGS = "machine_settings"
    GP_RACE = "gp_race"
    RESULTS = "results"
    GP_NEXT_COURSE = "gp_next_course"
    POST_GP = "post_gp"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CareerMenuTiming:
    """Frame cadence for runner-owned menu inputs."""

    start_hold_frames: int = 2
    start_settle_frames: int = 38
    menu_hold_frames: int = 8
    menu_settle_frames: int = 16
    engine_adjust_hold_frames: int = 2
    engine_adjust_settle_frames: int = 1
    engine_adjust_max_taps_per_burst: int = 25
    engine_ready_confirm_frames: int = 2
    difficulty_popup_open_settle_frames: int = 60
    result_continue_hold_frames: int = 4
    result_continue_settle_frames: int = 2
    max_advance_presses_per_phase: int = 12
    max_engine_adjust_taps: int = 128
    fresh_race_ready_frames: int = 2
    new_race_ready_time_ms: int = 1_000

    def __post_init__(self) -> None:
        values = (
            self.start_hold_frames,
            self.start_settle_frames,
            self.menu_hold_frames,
            self.menu_settle_frames,
            self.engine_adjust_hold_frames,
            self.engine_adjust_settle_frames,
            self.engine_adjust_max_taps_per_burst,
            self.engine_ready_confirm_frames,
            self.difficulty_popup_open_settle_frames,
            self.result_continue_hold_frames,
            self.result_continue_settle_frames,
            self.max_advance_presses_per_phase,
            self.max_engine_adjust_taps,
            self.fresh_race_ready_frames,
            self.new_race_ready_time_ms,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Career menu timing values must be positive")


MENU_TIMING = CareerMenuTiming()


@dataclass(frozen=True, slots=True)
class GpMenuOrder:
    """Menu order for GP difficulty and cup selection."""

    difficulties: tuple[str, ...] = ("novice", "standard", "expert", "master")
    cups: tuple[str, ...] = ("jack", "queen", "king", "joker")

    def difficulty_down_count(self, difficulty: str) -> int:
        return self._index_of(self.difficulties, difficulty, label="difficulty")

    def cup_right_count(self, cup_id: str) -> int:
        return self._index_of(self.cups, cup_id, label="cup")

    @staticmethod
    def _index_of(values: tuple[str, ...], value: str, *, label: str) -> int:
        try:
            return values.index(value)
        except ValueError as exc:
            known = ", ".join(values)
            raise RuntimeError(f"unknown Career Mode {label} {value!r}; known: {known}") from exc


GP_MENU_ORDER = GpMenuOrder()


@dataclass(frozen=True, slots=True)
class RawMenuStep:
    """One semantic input pulse emitted by the Career Mode FSM."""

    menu_input: MenuInput
    frames: int
    phase: str

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if not self.phase:
            raise ValueError("phase must not be empty")
