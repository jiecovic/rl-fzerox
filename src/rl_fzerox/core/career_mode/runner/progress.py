# src/rl_fzerox/core/career_mode/runner/progress.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.runner.race import (
    SaveRaceExecutionPlan,
    build_save_race_execution_plan,
)
from rl_fzerox.core.career_mode.runner.save_file import (
    SaveRamRuntimeSession,
    persist_save_ram_for_store,
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


class CareerAttemptProgress:
    """Persist save progress and move between managed Career Mode attempts."""

    def __init__(
        self,
        *,
        store: CareerModeStore,
        save_game_id: str,
        attempt_id: str,
    ) -> None:
        self._store = store
        self._save_game_id = save_game_id
        self._attempt_id: str | None = attempt_id
        self._course_setups = self._store.list_save_course_setups(save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(save_game_id)

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

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._refresh_unlock_progress()
        if _target_succeeded(progress.targets, setup):
            return self._finish_and_advance(
                info=info,
                status="succeeded",
                failure_reason=None,
            )

        if _keeps_current_gp_attempt(info):
            return CareerProgressTransition(attempt_finished=False)

        return self._finish_and_advance(
            info=info,
            status="failed",
            failure_reason=_failure_reason(info),
        )

    def sync_post_terminal_success(
        self,
        *,
        session: SaveRamRuntimeSession,
        setup: CareerModeRaceSetupConfig,
        info: dict[str, object],
    ) -> CareerProgressTransition:
        if self._attempt_id is None:
            return CareerProgressTransition(attempt_finished=False)

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._refresh_unlock_progress()
        if not _target_succeeded(progress.targets, setup):
            return CareerProgressTransition(attempt_finished=False)
        return self._finish_and_advance(
            info=info,
            status="succeeded",
            failure_reason=None,
        )

    def apply_execution_plan(self, plan: SaveRaceExecutionPlan) -> None:
        self._attempt_id = plan.attempt_id
        self._course_setups = self._store.list_save_course_setups(self._save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(self._save_game_id)

    def _finish_and_advance(
        self,
        *,
        info: dict[str, object],
        status: SaveAttemptStatus,
        failure_reason: str | None,
    ) -> CareerProgressTransition:
        finished_attempt_id = self._attempt_id
        self._finish_attempt(info=info, status=status, failure_reason=failure_reason)
        return self._advance_after_finished_attempt(
            finished_attempt_id=finished_attempt_id,
            finished_status=status,
            finished_failure_reason=failure_reason,
        )

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
        finished_attempt_id: str | None,
        finished_status: SaveAttemptStatus,
        finished_failure_reason: str | None,
    ) -> CareerProgressTransition:
        progress = self._refresh_unlock_progress()
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
            )

        next_attempt = self._store.start_next_save_attempt(self._save_game_id)
        context = self._store.get_save_attempt_execution_context(next_attempt.id)
        if context is None:
            raise RuntimeError(
                f"save attempt disappeared before Career Mode could continue: {next_attempt.id}"
            )
        return CareerProgressTransition(
            attempt_finished=True,
            next_plan=build_save_race_execution_plan(context),
            finished_attempt_id=finished_attempt_id,
            finished_status=finished_status,
            finished_failure_reason=finished_failure_reason,
        )

    def _refresh_unlock_progress(self) -> ManagedSaveUnlockProgress:
        self._unlock_progress = self._store.save_game_unlock_progress(self._save_game_id)
        return self._unlock_progress


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


def _failure_reason(info: dict[str, object]) -> str:
    reason = info.get("termination_reason")
    return reason if isinstance(reason, str) and reason else "race ended before cup clear"


def _keeps_current_gp_attempt(info: dict[str, object]) -> bool:
    reason = info.get("termination_reason")
    return reason in {"finished", "crashed", "retired"}


def _positive_int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None
