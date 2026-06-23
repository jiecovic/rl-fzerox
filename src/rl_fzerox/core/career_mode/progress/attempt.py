# src/rl_fzerox/core/career_mode/progress/attempt.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.execution.race import (
    SaveRaceExecutionPlan,
    build_save_race_execution_plan,
)
from rl_fzerox.core.career_mode.execution.save_file import (
    SaveRamRuntimeSession,
    persist_save_ram_for_store,
)
from rl_fzerox.core.career_mode.navigation import (
    POST_GP_RECORDING_END_MODES,
)
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, CourseInfo
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    SaveAttemptStatus,
    SaveGameStatus,
)
from rl_fzerox.core.runtime_info import optional_int_info, optional_str_info
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


class CareerModeStore(Protocol):
    """Manager operations the Career runner mutates while progressing a save."""

    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None: ...

    def list_save_course_setups(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveCourseSetup, ...]: ...

    def save_game_unlock_progress(
        self,
        save_game_id: str,
    ) -> ManagedSaveUnlockProgress: ...

    def finish_save_attempt(
        self,
        *,
        attempt_id: str,
        status: SaveAttemptStatus,
        finish_position: int | None = None,
        finish_time_s: float | None = None,
        failure_reason: str | None = None,
    ) -> ManagedSaveAttempt | None: ...

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> object | None: ...

    def start_next_save_attempt(self, save_game_id: str) -> ManagedSaveAttempt: ...

    def start_target_save_attempt(
        self,
        save_game_id: str,
        *,
        target_kind: str,
        difficulty: str,
        cup_id: str,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt: ...

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None: ...


@dataclass(frozen=True, slots=True)
class CareerProgressTransition:
    attempt_finished: bool
    next_plan: SaveRaceExecutionPlan | None = None
    finished_attempt_id: str | None = None
    finished_status: SaveAttemptStatus | None = None
    finished_failure_reason: str | None = None
    recording_info: dict[str, object] | None = None
    reset_emulator: bool = False


class CareerAttemptProgress:
    """Persist save progress and move between managed Career Mode attempts."""

    def __init__(
        self,
        *,
        store: CareerModeStore,
        save_game_id: str,
        attempt_id: str,
        single_target: bool = False,
        perfect_run: bool = False,
        target_clear_goal: int = 0,
    ) -> None:
        self._store = store
        self._save_game_id = save_game_id
        self._attempt_id: str | None = attempt_id
        self._single_target = single_target
        self._perfect_run = perfect_run
        self._target_clear_goal = target_clear_goal
        self._successful_target_clears = 0
        self._course_setups = self._store.list_save_course_setups(save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(save_game_id)
        self._initial_unlock_progress = self._unlock_progress
        self._observed_target_terminal_success = False
        self._observed_gp_final_rank: int | None = None
        self._observed_gp_points: int | None = None

    @property
    def attempt_id(self) -> str | None:
        return self._attempt_id

    @property
    def course_setups(self) -> tuple[ManagedSaveCourseSetup, ...]:
        return self._course_setups

    @property
    def unlock_progress(self) -> ManagedSaveUnlockProgress:
        return self._unlock_progress

    def handle_terminal_race(
        self,
        *,
        session: SaveRamRuntimeSession,
        setup: CareerModeRaceSetupConfig,
        info: dict[str, object],
    ) -> CareerProgressTransition:
        if self._attempt_id is None:
            return CareerProgressTransition(attempt_finished=False)

        if _target_terminal_succeeded(info, setup):
            self._observed_target_terminal_success = True

        if self._perfect_run and _is_failed_terminal_result(info):
            return self._finish_and_advance(
                info=info,
                setup=setup,
                status="failed",
                failure_reason=f"perfect run reset after {_failure_reason(info)}",
                reset_emulator=True,
            )

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._refresh_unlock_progress()
        if _target_succeeded(progress.targets, setup):
            return CareerProgressTransition(attempt_finished=False)

        if _keeps_current_gp_attempt(info):
            return CareerProgressTransition(attempt_finished=False)

        return self._finish_and_advance(
            info=info,
            setup=setup,
            status="failed",
            failure_reason=_failure_reason(info),
        )

    def sync_post_terminal_progress(
        self,
        *,
        session: SaveRamRuntimeSession,
        setup: CareerModeRaceSetupConfig,
        info: dict[str, object],
    ) -> CareerProgressTransition:
        if self._attempt_id is None:
            return CareerProgressTransition(attempt_finished=False)

        self._observe_gp_result(info)

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._refresh_unlock_progress()
        if _target_succeeded(progress.targets, setup):
            final_rank = self._observed_gp_final_rank or _gp_final_rank(info)
            if final_rank is not None and final_rank > 1:
                return self._finish_and_advance(
                    info=info,
                    setup=setup,
                    status="failed",
                    failure_reason=f"gp cup final rank {final_rank}",
                    reset_emulator=True,
                )
            if self._target_succeeded_before_attempt(setup) and final_rank != 1:
                if not self._observed_target_terminal_success:
                    return CareerProgressTransition(attempt_finished=False)
                if _is_post_gp_completion(info):
                    return self._finish_and_advance(
                        info=info,
                        setup=setup,
                        status="failed",
                        failure_reason="gp cup final rank missing",
                    )
                return CareerProgressTransition(attempt_finished=False)
            if not _is_post_gp_completion(info):
                return CareerProgressTransition(attempt_finished=False)
            return self._finish_and_advance(
                info=info,
                setup=setup,
                status="succeeded",
                failure_reason=None,
            )
        if _is_failed_gp_exit(info) and not self._observed_target_terminal_success:
            return self._finish_and_advance(
                info=info,
                setup=setup,
                status="failed",
                failure_reason="gp attempt returned to menu before cup clear",
            )
        return CareerProgressTransition(attempt_finished=False)

    def apply_execution_plan(self, plan: SaveRaceExecutionPlan) -> None:
        self._attempt_id = plan.attempt_id
        self._course_setups = self._store.list_save_course_setups(self._save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(self._save_game_id)
        self._initial_unlock_progress = self._unlock_progress
        self._observed_target_terminal_success = False
        self._observed_gp_final_rank = None
        self._observed_gp_points = None

    def observe_post_race_screen(
        self,
        *,
        info: dict[str, object],
        setup: CareerModeRaceSetupConfig,
    ) -> None:
        if _target_terminal_succeeded(info, setup):
            self._observed_target_terminal_success = True
        if self._observed_target_terminal_success:
            self._observe_gp_result(info)

    def _observe_gp_result(self, info: dict[str, object]) -> None:
        rank = _gp_final_rank(info)
        if rank is not None:
            self._observed_gp_final_rank = rank
        points = _gp_points(info)
        if points is not None:
            self._observed_gp_points = points

    def _finish_and_advance(
        self,
        *,
        info: dict[str, object],
        setup: CareerModeRaceSetupConfig,
        status: SaveAttemptStatus,
        failure_reason: str | None,
        reset_emulator: bool = False,
    ) -> CareerProgressTransition:
        finished_attempt_id = self._attempt_id
        finish_info = self._finish_info(info)
        self._finish_attempt(info=finish_info, status=status, failure_reason=failure_reason)
        return self._advance_after_finished_attempt(
            info=finish_info,
            setup=setup,
            finished_attempt_id=finished_attempt_id,
            finished_status=status,
            finished_failure_reason=failure_reason,
            reset_emulator=reset_emulator,
        )

    def _finish_info(self, info: dict[str, object]) -> dict[str, object]:
        finish_info = dict(info)
        if self._observed_gp_final_rank is not None:
            finish_info["career_mode_gp_final_rank"] = self._observed_gp_final_rank
        if self._observed_gp_points is not None:
            finish_info["career_mode_gp_points"] = self._observed_gp_points
        return finish_info

    def _finish_attempt(
        self,
        *,
        info: dict[str, object],
        status: SaveAttemptStatus,
        failure_reason: str | None,
    ) -> None:
        if self._attempt_id is None:
            return
        finish_time_ms = _positive_int_info(info, "race_time_ms")
        self._store.finish_save_attempt(
            attempt_id=self._attempt_id,
            status=status,
            finish_position=_positive_int_info(info, "position"),
            finish_time_s=(finish_time_ms / 1000.0 if finish_time_ms is not None else None),
            failure_reason=failure_reason,
        )

    def _advance_after_finished_attempt(
        self,
        *,
        info: dict[str, object],
        setup: CareerModeRaceSetupConfig,
        finished_attempt_id: str | None,
        finished_status: SaveAttemptStatus,
        finished_failure_reason: str | None,
        reset_emulator: bool,
    ) -> CareerProgressTransition:
        progress = self._refresh_unlock_progress()
        if self._single_target:
            if finished_status == "succeeded":
                self._successful_target_clears += 1
            if (
                finished_status == "succeeded"
                and self._target_clear_goal > 0
                and self._successful_target_clears >= self._target_clear_goal
            ):
                self._store.update_save_game_status(
                    save_game_id=self._save_game_id,
                    status="paused",
                )
                self._attempt_id = None
                return CareerProgressTransition(
                    attempt_finished=True,
                    finished_attempt_id=finished_attempt_id,
                    finished_status=finished_status,
                    finished_failure_reason=finished_failure_reason,
                    recording_info=dict(info),
                    reset_emulator=reset_emulator,
                )
            next_attempt = self._store.start_target_save_attempt(
                self._save_game_id,
                target_kind="clear_gp_cup",
                difficulty=setup.difficulty,
                cup_id=setup.cup_id,
                # Unlock targets are GP-cup clears. The resolved race setup may
                # carry a concrete course_id for policy setup; using it here
                # would turn a cup replay into a non-existent course target.
                course_id=None,
            )
            context = self._store.get_save_attempt_execution_context(next_attempt.id)
            if context is None:
                raise RuntimeError(
                    "save attempt disappeared before Career Mode could repeat selected target: "
                    f"{next_attempt.id}"
                )
            next_attempt_reset_emulator = reset_emulator or _is_post_gp_completion(info)
            return CareerProgressTransition(
                attempt_finished=True,
                next_plan=build_save_race_execution_plan(context),
                finished_attempt_id=finished_attempt_id,
                finished_status=finished_status,
                finished_failure_reason=finished_failure_reason,
                recording_info=dict(info),
                reset_emulator=next_attempt_reset_emulator,
            )
        if progress.next_target is None:
            self._store.update_save_game_status(
                save_game_id=self._save_game_id,
                status="finished",
            )
            self._attempt_id = None
            return CareerProgressTransition(
                attempt_finished=True,
                finished_attempt_id=finished_attempt_id,
                finished_status=finished_status,
                finished_failure_reason=finished_failure_reason,
                recording_info=dict(info),
                reset_emulator=reset_emulator,
            )

        next_attempt = self._store.start_next_save_attempt(self._save_game_id)
        context = self._store.get_save_attempt_execution_context(next_attempt.id)
        if context is None:
            raise RuntimeError(
                f"save attempt disappeared before Career Mode could continue: {next_attempt.id}"
            )
        next_attempt_reset_emulator = reset_emulator or _is_post_gp_completion(info)
        return CareerProgressTransition(
            attempt_finished=True,
            next_plan=build_save_race_execution_plan(context),
            finished_attempt_id=finished_attempt_id,
            finished_status=finished_status,
            finished_failure_reason=finished_failure_reason,
            recording_info=dict(info),
            reset_emulator=next_attempt_reset_emulator,
        )

    def _refresh_unlock_progress(self) -> ManagedSaveUnlockProgress:
        self._unlock_progress = self._store.save_game_unlock_progress(self._save_game_id)
        return self._unlock_progress

    def _target_succeeded_before_attempt(self, setup: CareerModeRaceSetupConfig) -> bool:
        return _target_succeeded(self._initial_unlock_progress.targets, setup)


def _target_succeeded(
    targets: Iterable[object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    return any(
        getattr(target, "status", None) == "succeeded"
        and getattr(target, "kind", None) == "clear_gp_cup"
        and getattr(target, "difficulty", None) == setup.difficulty
        and getattr(target, "cup_id", None) == setup.cup_id
        for target in targets
    )


def _target_terminal_succeeded(
    info: dict[str, object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    if info.get("termination_reason") != "finished":
        return False

    final_course = _target_final_course(setup)
    if final_course is None:
        return False
    return _info_matches_course(info, final_course) or _info_matches_final_cup_slot(
        info,
        setup,
    )


def _target_final_course(setup: CareerModeRaceSetupConfig) -> CourseInfo | None:
    courses = _target_cup_courses(setup)
    if not courses:
        return None
    return max(courses, key=lambda course: course.course_index)


def _target_cup_courses(setup: CareerModeRaceSetupConfig) -> tuple[CourseInfo, ...]:
    return tuple(
        sorted(
            (course for course in BUILT_IN_COURSES if course.cup == setup.cup_id),
            key=lambda course: course.course_index,
        )
    )


def _info_matches_course(info: dict[str, object], course: CourseInfo) -> bool:
    course_index = _int_info(info, "track_course_index")
    if course_index is None:
        course_index = _int_info(info, "course_index")
    if course_index == course.course_index:
        return True

    course_id = _str_info(info, "track_id")
    if course_id is None:
        course_id = _str_info(info, "course_id")
    return course_id in {course.id, course.ref}


def _info_matches_final_cup_slot(
    info: dict[str, object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    courses = _target_cup_courses(setup)
    if not courses:
        return False
    final_slot = len(courses) - 1

    # On live GP result handoff some native fields expose the cup-local slot
    # instead of the global built-in course index. Accept that only as terminal
    # final-course evidence; post-GP success/failure still owns the cup result.
    for key in ("career_mode_cup_course_index", "cup_course_index", "course_index"):
        if _int_info(info, key) == final_slot:
            return True
    return False


def _failure_reason(info: dict[str, object]) -> str:
    reason = info.get("termination_reason")
    return reason if isinstance(reason, str) and reason else "race ended before cup clear"


def _is_failed_terminal_result(info: dict[str, object]) -> bool:
    return info.get("termination_reason") in {"crashed", "retired"}


def _keeps_current_gp_attempt(info: dict[str, object]) -> bool:
    reason = info.get("termination_reason")
    return reason in {"finished", "crashed", "retired"}


def _is_post_gp_completion(info: dict[str, object]) -> bool:
    mode = info.get("game_mode")
    if not isinstance(mode, str) or not mode:
        mode = info.get("game_mode_name")
    if mode == "gp_end_cutscene":
        return info.get("career_mode_post_gp_cutscene_complete") is True
    return mode in POST_GP_RECORDING_END_MODES or mode in {
        "title",
        "main_menu",
        "course_select",
    }


def _is_failed_gp_exit(info: dict[str, object]) -> bool:
    mode = info.get("game_mode")
    if not isinstance(mode, str) or not mode:
        mode = info.get("game_mode_name")
    return mode in {
        "title",
        "main_menu",
        "course_select",
    }


def _positive_int_info(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key, float_values=False, minimum=1)


_GP_RESULT_MODES = {
    "gp_race",
    "results",
    "gp_end_cutscene",
}


def _gp_final_rank(info: dict[str, object]) -> int | None:
    explicit_rank = _positive_int_info(info, "career_mode_gp_final_rank")
    if explicit_rank is not None:
        return explicit_rank
    mode = info.get("game_mode")
    if not isinstance(mode, str) or not mode:
        mode = info.get("game_mode_name")
    # The native rank is derived from the live GP points table. Career progress
    # only asks for it while handling post-terminal GP flow, so these modes are
    # the windows where the table still belongs to the current cup attempt.
    if mode not in _GP_RESULT_MODES:
        return None
    return _positive_int_info(info, "gp_final_rank")


def _gp_points(info: dict[str, object]) -> int | None:
    explicit_points = _gp_points_value(info.get("career_mode_gp_points"))
    if explicit_points is not None:
        return explicit_points
    mode = info.get("game_mode")
    if not isinstance(mode, str) or not mode:
        mode = info.get("game_mode_name")
    # `Racer.points` remains authoritative from the race result through the GP
    # end cutscene; after menu/reset transitions it can be stale or reset.
    if mode not in _GP_RESULT_MODES:
        return None
    return _gp_points_value(info.get("gp_points"))


def _gp_points_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 999 else None


def _int_info(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key, float_values=False)


def _str_info(info: dict[str, object], key: str) -> str | None:
    return optional_str_info(info, key, non_empty=True)
