# src/rl_fzerox/core/career_mode/navigation/facts.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation.types import (
    MENU_TIMING,
    DifficultyPopupState,
    ObservedMenuScreen,
)
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.domain.engine_setting import ENGINE_SLIDER
from rl_fzerox.core.runtime_info import (
    bool_info,
    optional_float_info,
    optional_int_info,
    optional_str_info,
)

BUILT_IN_COURSES_BY_INDEX = {course.course_index: course for course in BUILT_IN_COURSES}
POST_GP_COMPLETION_MODES = frozenset(
    {
        "gp_end_cutscene",
        "skippable_credits",
        "unskippable_credits",
    }
)
POST_GP_RECORDING_END_MODES = frozenset(
    {
        "skippable_credits",
        "unskippable_credits",
    }
)


@dataclass(frozen=True, slots=True)
class MenuFacts:
    """Native menu/race facts consumed by the Career Mode FSM."""

    game_mode: str | None
    game_mode_raw: int | None
    selected_mode_raw: int | None
    difficulty_state_raw: int | None
    difficulty_cursor_raw: int | None
    transition_state_raw: int | None
    current_ghost_type_raw: int | None
    queued_game_mode_raw: int | None
    course_index: int | None
    difficulty_raw: int | None
    camera_setting: str | None
    race_intro_timer: int | None
    race_time_ms: float | None
    engine_setting_raw: int | None
    completion_fraction: float | None
    completed_laps: int | None
    total_laps: int | None
    terminal_reason: str | None
    finished: bool
    retired: bool
    crashed: bool
    entered_finished: bool
    entered_retired: bool
    entered_crashed: bool

    @classmethod
    def from_info(cls, info: dict[str, object]) -> MenuFacts:
        entered_finished = bool_info(info, "entered_finished")
        entered_retired = bool_info(info, "entered_retired")
        entered_crashed = bool_info(info, "entered_crashed")
        return cls(
            game_mode=game_mode(info),
            game_mode_raw=_int_info(info, "game_mode_raw"),
            selected_mode_raw=_int_info(info, "menu_selected_mode_raw"),
            difficulty_state_raw=_int_info(info, "menu_difficulty_state_raw"),
            difficulty_cursor_raw=_int_info(info, "menu_difficulty_cursor_raw"),
            transition_state_raw=_int_info(info, "menu_transition_state_raw"),
            current_ghost_type_raw=_int_info(info, "menu_current_ghost_type_raw"),
            queued_game_mode_raw=_int_info(info, "queued_game_mode_raw"),
            course_index=_int_info(info, "course_index"),
            difficulty_raw=_int_info(info, "difficulty_raw"),
            camera_setting=camera_setting(info),
            race_intro_timer=_int_info(info, "race_intro_timer"),
            race_time_ms=_number_info(info, "race_time_ms"),
            engine_setting_raw=_int_info(info, "engine_setting_raw_value_ram"),
            completion_fraction=_number_info(info, "episode_completion_fraction"),
            completed_laps=_int_info(info, "race_laps_completed"),
            total_laps=_int_info(info, "total_lap_count"),
            terminal_reason=_str_info(info, "termination_reason"),
            finished=bool_info(info, "finished") or entered_finished,
            retired=bool_info(info, "retired") or entered_retired,
            crashed=bool_info(info, "crashed") or entered_crashed,
            entered_finished=entered_finished,
            entered_retired=entered_retired,
            entered_crashed=entered_crashed,
        )

    @property
    def is_title(self) -> bool:
        return self.game_mode == "title"

    @property
    def is_mode_select(self) -> bool:
        return self.game_mode == "main_menu"

    @property
    def has_difficulty_popup(self) -> bool:
        return self.is_mode_select and self.selected_gp_mode and self.difficulty_state_raw == 1

    @property
    def submitted_difficulty_popup(self) -> bool:
        return self.is_mode_select and self.selected_gp_mode and self.difficulty_state_raw == 2

    @property
    def is_game_mode_transition(self) -> bool:
        if self.transition_state_raw is not None:
            return self.transition_state_raw != 0
        return False

    @property
    def selected_gp_mode(self) -> bool:
        return self.selected_mode_raw in (None, 0)

    @property
    def is_course_select(self) -> bool:
        return self.game_mode == "course_select"

    @property
    def is_machine_select(self) -> bool:
        return self.game_mode == "machine_select"

    @property
    def is_machine_settings(self) -> bool:
        return self.game_mode in {"machine_settings", "gp_race_next_machine_settings"}

    @property
    def in_gp_race(self) -> bool:
        return self.game_mode == "gp_race"

    @property
    def is_gp_end_cutscene(self) -> bool:
        return self.game_mode == "gp_end_cutscene"

    @property
    def is_gp_next_course_screen(self) -> bool:
        return self.game_mode == "gp_race_next_course"

    @property
    def is_gp_result_screen(self) -> bool:
        return self.game_mode == "results"

    @property
    def is_post_gp_screen(self) -> bool:
        return self.game_mode in POST_GP_COMPLETION_MODES

    @property
    def course_select_cup_index(self) -> int | None:
        if self.course_index is None or self.course_index < 0:
            return None
        return self.course_index // 6

    @property
    def terminal_race_result(self) -> bool:
        return (
            self.terminal_reason in {"finished", "retired", "crashed"}
            or self.entered_finished
            or self.entered_retired
            or self.entered_crashed
        )

    @property
    def completed_race_laps(self) -> bool:
        if self.completed_laps is None or self.total_laps is None:
            return False
        return self.total_laps > 0 and self.completed_laps >= self.total_laps

    @property
    def fresh_race_ready_for_policy(self) -> bool:
        """Return true once a post-result continuation reaches a new race."""

        if self.terminal_race_result:
            return False
        return self.has_fresh_race_shape

    @property
    def has_fresh_race_shape(self) -> bool:
        """Return true for the first countdown frames of a new GP race."""

        if self.course_index is None:
            return False
        if self.completed_laps is None or self.total_laps is None:
            return False
        if self.total_laps <= 0 or self.completed_laps != 0:
            return False
        if self.completed_race_laps:
            return False
        if self.race_intro_timer is None or self.race_intro_timer <= 0:
            return False
        if self.completion_fraction is not None and self.completion_fraction >= 0.95:
            return False
        if self.race_time_ms is None:
            return False
        return self.race_time_ms <= MENU_TIMING.new_race_ready_time_ms

    @property
    def engine_setting_raw_value(self) -> int | None:
        if self.engine_setting_raw is None:
            return None
        if not ENGINE_SLIDER.min_step <= self.engine_setting_raw <= ENGINE_SLIDER.max_step:
            return None
        return self.engine_setting_raw


def observed_menu_screen(
    facts: MenuFacts,
    *,
    difficulty_popup_state: DifficultyPopupState,
) -> ObservedMenuScreen:
    """Classify the current screen from RAM-backed telemetry where available."""

    if facts.is_title:
        return ObservedMenuScreen.TITLE
    if facts.is_course_select:
        return ObservedMenuScreen.COURSE_SELECT
    if facts.is_machine_select:
        return ObservedMenuScreen.MACHINE_SELECT
    if facts.is_machine_settings:
        return ObservedMenuScreen.MACHINE_SETTINGS
    if facts.is_gp_result_screen:
        return ObservedMenuScreen.RESULTS
    if facts.is_gp_next_course_screen:
        return ObservedMenuScreen.GP_NEXT_COURSE
    if facts.is_post_gp_screen:
        return ObservedMenuScreen.POST_GP
    if facts.in_gp_race:
        return ObservedMenuScreen.GP_RACE
    if facts.is_mode_select:
        if facts.has_difficulty_popup:
            return ObservedMenuScreen.DIFFICULTY_POPUP
        if facts.submitted_difficulty_popup:
            return ObservedMenuScreen.DIFFICULTY_CONFIRM
        if difficulty_popup_state is DifficultyPopupState.OPENING:
            return ObservedMenuScreen.TRANSITION
        if facts.is_game_mode_transition:
            return ObservedMenuScreen.TRANSITION
        if facts.selected_gp_mode:
            return ObservedMenuScreen.MAIN_MENU_GP
        return ObservedMenuScreen.MAIN_MENU_OTHER
    if facts.is_game_mode_transition:
        return ObservedMenuScreen.TRANSITION
    return ObservedMenuScreen.UNKNOWN


def game_mode(info: dict[str, object]) -> str | None:
    return _str_info(info, "game_mode") or _str_info(info, "game_mode_name")


def is_title_mode(info: dict[str, object]) -> bool:
    return game_mode(info) == "title"


def is_mode_select(info: dict[str, object]) -> bool:
    return game_mode(info) == "main_menu"


def course_select_cup_index(info: dict[str, object]) -> int | None:
    course_index = _int_info(info, "course_index")
    if course_index is None:
        return None
    if course_index < 0:
        return None
    return course_index // 6


def course_id_from_info(info: dict[str, object]) -> str | None:
    course_index = _int_info(info, "course_index")
    if course_index is None:
        return None
    course = BUILT_IN_COURSES_BY_INDEX.get(course_index)
    if course is None:
        return None
    return course.id


def in_gp_race(info: dict[str, object]) -> bool:
    return game_mode(info) == "gp_race"


def camera_setting(info: dict[str, object]) -> str | None:
    return _str_info(info, "camera_setting") or _str_info(info, "camera_setting_name")


def _int_info(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key)


def _str_info(info: dict[str, object], key: str) -> str | None:
    return optional_str_info(info, key, non_empty=True)


def _number_info(info: dict[str, object], key: str) -> float | None:
    return optional_float_info(info, key)
