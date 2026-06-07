# src/rl_fzerox/core/career_mode/runner/menu.py
"""Menu state machines for Career Mode runner control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


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
    WAIT_FOR_GP_RACE = "wait_for_gp_race"


class DifficultyPopupState(StrEnum):
    """Controller-owned state for the GP difficulty overlay."""

    CLOSED = "closed"
    OPEN = "open"
    SUBMITTED = "submitted"


class ObservedMenuScreen(StrEnum):
    """Visible game screen inferred from native telemetry."""

    TITLE = "title"
    MAIN_MENU_GP = "main_menu_gp"
    MAIN_MENU_OTHER = "main_menu_other"
    DIFFICULTY_POPUP = "difficulty_popup"
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
    difficulty_popup_open_settle_frames: int = 60
    result_continue_hold_frames: int = 4
    result_continue_settle_frames: int = 2
    max_start_presses_per_phase: int = 12
    max_engine_adjust_taps: int = 128
    fresh_race_ready_frames: int = 2
    new_race_ready_time_ms: int = 1_000

    def __post_init__(self) -> None:
        values = (
            self.start_hold_frames,
            self.start_settle_frames,
            self.menu_hold_frames,
            self.menu_settle_frames,
            self.difficulty_popup_open_settle_frames,
            self.result_continue_hold_frames,
            self.result_continue_settle_frames,
            self.max_start_presses_per_phase,
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
BUILT_IN_COURSES_BY_INDEX = {course.course_index: course for course in BUILT_IN_COURSES}


@dataclass(frozen=True, slots=True)
class CareerMenuDefaults:
    """Game menu defaults Career Mode can rely on before menu editing exists."""

    engine_setting_raw_value: int = 50


CAREER_MENU_DEFAULTS = CareerMenuDefaults()


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


@dataclass(frozen=True, slots=True)
class MenuFacts:
    """Native menu/race facts consumed by the Career Mode FSM."""

    game_mode: str | None
    game_mode_raw: int | None
    selected_mode_raw: int | None
    transition_state_raw: int | None
    current_ghost_type_raw: int | None
    queued_game_mode_raw: int | None
    course_index: int | None
    difficulty_raw: int | None
    camera_setting: str | None
    race_intro_timer: int | None
    race_time_ms: float | None
    engine_setting_percent: float | None
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
        entered_finished = info.get("entered_finished") is True
        entered_retired = info.get("entered_retired") is True
        entered_crashed = info.get("entered_crashed") is True
        return cls(
            game_mode=game_mode(info),
            game_mode_raw=_int_info(info, "game_mode_raw"),
            selected_mode_raw=_int_info(info, "menu_selected_mode_raw"),
            transition_state_raw=_int_info(info, "menu_transition_state_raw"),
            current_ghost_type_raw=_int_info(info, "menu_current_ghost_type_raw"),
            queued_game_mode_raw=_int_info(info, "queued_game_mode_raw"),
            course_index=_int_info(info, "course_index"),
            difficulty_raw=_int_info(info, "difficulty_raw"),
            camera_setting=camera_setting(info),
            race_intro_timer=_int_info(info, "race_intro_timer"),
            race_time_ms=_number_info(info, "race_time_ms"),
            engine_setting_percent=_number_info(info, "engine_setting_percent_ram"),
            completion_fraction=_number_info(info, "episode_completion_fraction"),
            completed_laps=_int_info(info, "race_laps_completed"),
            total_laps=_int_info(info, "total_lap_count"),
            terminal_reason=_str_info(info, "termination_reason"),
            finished=info.get("finished") is True or entered_finished,
            retired=info.get("retired") is True or entered_retired,
            crashed=info.get("crashed") is True or entered_crashed,
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
    def is_skippable_post_gp_screen(self) -> bool:
        return self.game_mode in {"gp_end_cutscene", "skippable_credits"}

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
        if self.engine_setting_percent is None:
            return None
        if not 0.0 <= self.engine_setting_percent <= 100.0:
            return None
        return round(self.engine_setting_percent)


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
    if facts.is_skippable_post_gp_screen:
        return ObservedMenuScreen.POST_GP
    if facts.in_gp_race:
        return ObservedMenuScreen.GP_RACE
    if facts.is_mode_select:
        if difficulty_popup_state is DifficultyPopupState.OPEN and facts.selected_gp_mode:
            return ObservedMenuScreen.DIFFICULTY_POPUP
        if difficulty_popup_state is DifficultyPopupState.SUBMITTED and facts.selected_gp_mode:
            return ObservedMenuScreen.TRANSITION
        if facts.is_game_mode_transition:
            return ObservedMenuScreen.TRANSITION
        if facts.selected_gp_mode:
            return ObservedMenuScreen.MAIN_MENU_GP
        return ObservedMenuScreen.MAIN_MENU_OTHER
    if facts.is_game_mode_transition:
        return ObservedMenuScreen.TRANSITION
    return ObservedMenuScreen.UNKNOWN


def select_difficulty_steps(
    _setup: CareerModeRaceSetupConfig,
) -> tuple[RawMenuStep, ...]:
    return tap_steps(
        MenuInput.ACCEPT,
        hold_frames=MENU_TIMING.menu_hold_frames,
        settle_frames=MENU_TIMING.difficulty_popup_open_settle_frames,
        phase="select_difficulty:open",
    )


def select_open_difficulty_steps(
    setup: CareerModeRaceSetupConfig,
) -> tuple[RawMenuStep, ...]:
    steps: list[RawMenuStep] = []
    for tap_index in range(GP_MENU_ORDER.difficulty_down_count(setup.difficulty)):
        steps.extend(
            tap_steps(
                MenuInput.DOWN,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase=f"select_difficulty:down:{tap_index + 1}",
            )
        )
    steps.extend(
        tap_steps(
            MenuInput.ACCEPT,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="select_difficulty:accept",
        )
    )
    return tuple(steps)


def machine_select_steps(
    setup: CareerModeRaceSetupConfig,
) -> tuple[RawMenuStep, ...]:
    steps: list[RawMenuStep] = []
    for tap_index in range(setup.machine_select_row):
        steps.extend(
            tap_steps(
                MenuInput.DOWN,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase=f"select_machine:down:{tap_index + 1}",
            )
        )
    for tap_index in range(setup.machine_select_column):
        steps.extend(
            tap_steps(
                MenuInput.RIGHT,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase=f"select_machine:right:{tap_index + 1}",
            )
        )
    return tuple(steps)


def continue_after_race_step(press_index: int) -> RawMenuStep:
    """Return one post-race continuation pulse after a terminal edge."""

    return raw_step(
        MenuInput.ACCEPT,
        MENU_TIMING.result_continue_hold_frames,
        phase=f"continue_after_race:accept:{press_index + 1}",
    )


def continue_next_course_step() -> RawMenuStep:
    """Advance the GP next-course screen to machine settings."""

    return raw_step(
        MenuInput.ACCEPT,
        MENU_TIMING.menu_hold_frames,
        phase="continue_after_race:next_course_accept",
    )


def tap_steps(
    menu_input: MenuInput,
    *,
    hold_frames: int,
    settle_frames: int,
    phase: str,
) -> tuple[RawMenuStep, ...]:
    return (
        raw_step(menu_input, hold_frames, phase=phase),
        raw_step(MenuInput.NEUTRAL, settle_frames, phase=f"{phase}:settle"),
    )


def raw_step(
    menu_input: MenuInput,
    frames: int,
    *,
    phase: str,
) -> RawMenuStep:
    return RawMenuStep(menu_input=menu_input, frames=frames, phase=phase)


def phase_from_step(step: RawMenuStep) -> CareerPhase:
    if step.phase.startswith("select_difficulty"):
        return CareerPhase.SELECT_DIFFICULTY
    if step.phase.startswith("enter_course_select"):
        return CareerPhase.ENTER_COURSE_SELECT
    if step.phase.startswith("select_cup"):
        return CareerPhase.SELECT_CUP
    if step.phase.startswith("enter_machine_select"):
        return CareerPhase.ENTER_MACHINE_SELECT
    if step.phase.startswith("continue_after_race"):
        return CareerPhase.CONTINUE_AFTER_RACE
    if step.phase.startswith("select_machine"):
        return CareerPhase.SELECT_MACHINE
    if step.phase.startswith("enter_machine_settings"):
        return CareerPhase.ENTER_MACHINE_SETTINGS
    if step.phase.startswith("apply_engine"):
        return CareerPhase.APPLY_ENGINE
    if step.phase.startswith("enter_race"):
        return CareerPhase.ENTER_RACE
    return CareerPhase.BOOT_TO_DIFFICULTY


def game_mode(info: dict[str, object]) -> str | None:
    value = info.get("game_mode")
    return value if isinstance(value, str) and value else None


def is_title_mode(info: dict[str, object]) -> bool:
    return game_mode(info) == "title"


def is_mode_select(info: dict[str, object]) -> bool:
    return game_mode(info) == "main_menu"


def course_select_cup_index(info: dict[str, object]) -> int | None:
    course_index = info.get("course_index")
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return None
    if course_index < 0:
        return None
    return course_index // 6


def course_id_from_info(info: dict[str, object]) -> str | None:
    course_index = info.get("course_index")
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return None
    course = BUILT_IN_COURSES_BY_INDEX.get(course_index)
    if course is None:
        return None
    return course.id


def in_gp_race(info: dict[str, object]) -> bool:
    return info.get("game_mode") == "gp_race"


def camera_setting(info: dict[str, object]) -> str | None:
    value = info.get("camera_setting")
    if not isinstance(value, str) or not value:
        value = info.get("camera_setting_name")
    return value if isinstance(value, str) and value else None


def _int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _str_info(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _number_info(info: dict[str, object], key: str) -> float | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
