# src/rl_fzerox/core/career_mode/progress/attempt.py
"""Managed save-attempt progress transitions for Career Mode GP cups.

This module persists save RAM, refreshes unlock progress from the manager store,
and returns explicit transition objects for the controller to apply.
"""

from __future__ import annotations

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
from rl_fzerox.core.career_mode.progress.outcomes import (
    failure_reason,
    gp_final_rank,
    gp_points,
    is_failed_gp_exit,
    is_failed_terminal_result,
    is_post_gp_completion,
    keeps_current_gp_attempt,
    positive_int_info,
    target_succeeded,
    target_terminal_succeeded,
)
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    SaveAttemptStatus,
    SaveGameStatus,
)
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

        if target_terminal_succeeded(info, setup):
            self._observed_target_terminal_success = True

        if self._perfect_run and is_failed_terminal_result(info):
            return self._finish_and_advance(
                info=info,
                setup=setup,
                status="failed",
                failure_reason=f"perfect run reset after {failure_reason(info)}",
                reset_emulator=True,
            )

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._refresh_unlock_progress()
        if target_succeeded(progress.targets, setup):
            return CareerProgressTransition(attempt_finished=False)

        if keeps_current_gp_attempt(info):
            return CareerProgressTransition(attempt_finished=False)

        return self._finish_and_advance(
            info=info,
            setup=setup,
            status="failed",
            failure_reason=failure_reason(info),
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
        if target_succeeded(progress.targets, setup):
            final_rank = self._observed_gp_final_rank or gp_final_rank(info)
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
                if is_post_gp_completion(info):
                    return self._finish_and_advance(
                        info=info,
                        setup=setup,
                        status="failed",
                        failure_reason="gp cup final rank missing",
                    )
                return CareerProgressTransition(attempt_finished=False)
            if not is_post_gp_completion(info):
                return CareerProgressTransition(attempt_finished=False)
            return self._finish_and_advance(
                info=info,
                setup=setup,
                status="succeeded",
                failure_reason=None,
            )
        if is_failed_gp_exit(info) and not self._observed_target_terminal_success:
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
        if target_terminal_succeeded(info, setup):
            self._observed_target_terminal_success = True
        if self._observed_target_terminal_success:
            self._observe_gp_result(info)

    def _observe_gp_result(self, info: dict[str, object]) -> None:
        rank = gp_final_rank(info)
        if rank is not None:
            self._observed_gp_final_rank = rank
        points = gp_points(info)
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
        finish_time_ms = positive_int_info(info, "race_time_ms")
        self._store.finish_save_attempt(
            attempt_id=self._attempt_id,
            status=status,
            finish_position=positive_int_info(info, "position"),
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
            next_attempt_reset_emulator = reset_emulator or is_post_gp_completion(info)
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
        next_attempt_reset_emulator = reset_emulator or is_post_gp_completion(info)
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
        return target_succeeded(self._initial_unlock_progress.targets, setup)
