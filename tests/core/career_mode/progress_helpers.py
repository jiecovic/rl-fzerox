# tests/core/career_mode/progress_helpers.py
"""Shared fakes and builders for Career Mode progress tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.career_mode.course_setup import CourseSetupTarget
from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
    SaveAttemptStatus,
    SaveGameStatus,
)
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


@dataclass(slots=True)
class _Emulator:
    def read_save_ram(self) -> bytes:
        return b"save-ram"

    def write_save_ram(self, data: bytes) -> None:
        raise AssertionError("progress should only persist save RAM in this test")


@dataclass(frozen=True, slots=True)
class _Session:
    emulator: _Emulator = field(default_factory=_Emulator)


class _Store:
    def __init__(self, tmp_path: Path) -> None:
        self.save_game = ManagedSaveGame(
            id="save",
            name="Save",
            status="running",
            save_path=tmp_path / "fzerox.srm",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        self.finished_attempts: list[tuple[str, SaveAttemptStatus, str | None]] = []
        self.started_next_attempt_count = 0

    def get_save_game(self, save_game_id: str) -> ManagedSaveGame | None:
        return self.save_game if save_game_id == self.save_game.id else None

    def list_save_course_setups(
        self,
        save_game_id: str,
    ) -> tuple[ManagedSaveCourseSetup, ...]:
        return ()

    def save_game_unlock_progress(self, save_game_id: str) -> ManagedSaveUnlockProgress:
        return ManagedSaveUnlockProgress(
            inspection_status="inspected",
            completed_count=0,
            total_count=1,
            unlocked_vehicle_count=0,
            unlocked_vehicle_ids=(),
            next_target=ManagedSaveUnlockTarget(
                sequence_index=0,
                kind="clear_gp_cup",
                status="pending",
                label="Expert Jack Cup",
                difficulty="expert",
                cup_id="jack",
            ),
            targets=(
                ManagedSaveUnlockTarget(
                    sequence_index=0,
                    kind="clear_gp_cup",
                    status="pending",
                    label="Expert Jack Cup",
                    difficulty="expert",
                    cup_id="jack",
                ),
            ),
        )

    def finish_save_attempt(
        self,
        *,
        attempt_id: str,
        status: SaveAttemptStatus,
        finish_position: int | None = None,
        finish_time_s: float | None = None,
        failure_reason: str | None = None,
    ) -> ManagedSaveAttempt | None:
        self.finished_attempts.append((attempt_id, status, failure_reason))
        raise AssertionError("crashed GP races must not finish the save attempt")

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> object | None:
        raise AssertionError("save game status should not change")

    def start_next_save_attempt(self, save_game_id: str) -> ManagedSaveAttempt:
        self.started_next_attempt_count += 1
        raise AssertionError("crashed GP races must not start the next attempt")

    def start_target_save_attempt(
        self,
        save_game_id: str,
        *,
        target_kind: str,
        difficulty: str,
        cup_id: str,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        del save_game_id, target_kind, difficulty, cup_id, course_id
        raise AssertionError("crashed GP races must not start a target retry attempt")

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        raise AssertionError("no execution context should be requested")


class _SingleTargetCompletionStore(_Store):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.tmp_path = tmp_path
        self.status_updates: list[SaveGameStatus] = []
        self.progress_reads = 0
        self.next_attempt: ManagedSaveAttempt | None = None

    def save_game_unlock_progress(self, save_game_id: str) -> ManagedSaveUnlockProgress:
        self.progress_reads += 1
        target_status = "succeeded" if self.progress_reads >= 2 else "pending"
        return ManagedSaveUnlockProgress(
            inspection_status="inspected",
            completed_count=1 if target_status == "succeeded" else 0,
            total_count=2,
            unlocked_vehicle_count=0,
            unlocked_vehicle_ids=(),
            next_target=ManagedSaveUnlockTarget(
                sequence_index=0 if target_status == "pending" else 1,
                kind="clear_gp_cup",
                status="pending",
                label=("Expert Jack Cup" if target_status == "pending" else "Expert Queen Cup"),
                difficulty="expert",
                cup_id="jack" if target_status == "pending" else "queen",
            ),
            targets=(
                ManagedSaveUnlockTarget(
                    sequence_index=0,
                    kind="clear_gp_cup",
                    status=target_status,
                    label="Expert Jack Cup",
                    difficulty="expert",
                    cup_id="jack",
                ),
                ManagedSaveUnlockTarget(
                    sequence_index=1,
                    kind="clear_gp_cup",
                    status="pending",
                    label="Expert Queen Cup",
                    difficulty="expert",
                    cup_id="queen",
                ),
            ),
        )

    def finish_save_attempt(
        self,
        *,
        attempt_id: str,
        status: SaveAttemptStatus,
        finish_position: int | None = None,
        finish_time_s: float | None = None,
        failure_reason: str | None = None,
    ) -> ManagedSaveAttempt | None:
        del finish_position, finish_time_s
        self.finished_attempts.append((attempt_id, status, failure_reason))
        return None

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> object | None:
        self.status_updates.append(status)
        return None

    def start_next_save_attempt(self, save_game_id: str) -> ManagedSaveAttempt:
        self.started_next_attempt_count += 1
        raise AssertionError("single-target success must not start the next attempt")

    def start_target_save_attempt(
        self,
        save_game_id: str,
        *,
        target_kind: str,
        difficulty: str,
        cup_id: str,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        self.started_next_attempt_count += 1
        self.next_attempt = ManagedSaveAttempt(
            id=f"attempt-{self.started_next_attempt_count + 1}",
            save_game_id=save_game_id,
            status="running",
            started_at="2026-01-01T00:01:00Z",
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )
        return self.next_attempt

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        if self.next_attempt is None or attempt_id != self.next_attempt.id:
            return None
        return _execution_context(
            save_game=self.save_game,
            attempt=self.next_attempt,
            tmp_path=self.tmp_path,
        )


class _ReplayTargetStore(_SingleTargetCompletionStore):
    def save_game_unlock_progress(self, save_game_id: str) -> ManagedSaveUnlockProgress:
        return ManagedSaveUnlockProgress(
            inspection_status="inspected",
            completed_count=1,
            total_count=2,
            unlocked_vehicle_count=0,
            unlocked_vehicle_ids=(),
            next_target=ManagedSaveUnlockTarget(
                sequence_index=1,
                kind="clear_gp_cup",
                status="pending",
                label="Expert Queen Cup",
                difficulty="expert",
                cup_id="queen",
            ),
            targets=(
                ManagedSaveUnlockTarget(
                    sequence_index=0,
                    kind="clear_gp_cup",
                    status="succeeded",
                    label="Expert Jack Cup",
                    difficulty="expert",
                    cup_id="jack",
                ),
                ManagedSaveUnlockTarget(
                    sequence_index=1,
                    kind="clear_gp_cup",
                    status="pending",
                    label="Expert Queen Cup",
                    difficulty="expert",
                    cup_id="queen",
                ),
            ),
        )


class _FailedRetryStore(_Store):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.tmp_path = tmp_path
        self.status_updates: list[SaveGameStatus] = []
        self.next_attempt: ManagedSaveAttempt | None = None

    def save_game_unlock_progress(self, save_game_id: str) -> ManagedSaveUnlockProgress:
        return ManagedSaveUnlockProgress(
            inspection_status="inspected",
            completed_count=0,
            total_count=1,
            unlocked_vehicle_count=0,
            unlocked_vehicle_ids=(),
            next_target=ManagedSaveUnlockTarget(
                sequence_index=0,
                kind="clear_gp_cup",
                status="pending",
                label="Expert Jack Cup",
                difficulty="expert",
                cup_id="jack",
            ),
            targets=(
                ManagedSaveUnlockTarget(
                    sequence_index=0,
                    kind="clear_gp_cup",
                    status="pending",
                    label="Expert Jack Cup",
                    difficulty="expert",
                    cup_id="jack",
                ),
            ),
        )

    def finish_save_attempt(
        self,
        *,
        attempt_id: str,
        status: SaveAttemptStatus,
        finish_position: int | None = None,
        finish_time_s: float | None = None,
        failure_reason: str | None = None,
    ) -> ManagedSaveAttempt | None:
        del finish_position, finish_time_s
        self.finished_attempts.append((attempt_id, status, failure_reason))
        return None

    def update_save_game_status(
        self,
        *,
        save_game_id: str,
        status: SaveGameStatus,
    ) -> object | None:
        self.status_updates.append(status)
        return None

    def start_next_save_attempt(self, save_game_id: str) -> ManagedSaveAttempt:
        self.started_next_attempt_count += 1
        self.next_attempt = ManagedSaveAttempt(
            id="attempt-2",
            save_game_id=save_game_id,
            status="running",
            started_at="2026-01-01T00:01:00Z",
            target_kind="clear_gp_cup",
            difficulty="expert",
            cup_id="jack",
        )
        return self.next_attempt

    def start_target_save_attempt(
        self,
        save_game_id: str,
        *,
        target_kind: str,
        difficulty: str,
        cup_id: str,
        course_id: str | None = None,
    ) -> ManagedSaveAttempt:
        del target_kind, difficulty, cup_id, course_id
        return self.start_next_save_attempt(save_game_id)

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        if self.next_attempt is None or attempt_id != self.next_attempt.id:
            return None
        return _execution_context(
            save_game=self.save_game,
            attempt=self.next_attempt,
            tmp_path=self.tmp_path,
        )


def _race_setup(course_id: str | None = None) -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="expert",
        cup_id="jack",
        course_id=course_id,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_raw_value=50,
    )


def _execution_context(
    *,
    save_game: ManagedSaveGame,
    attempt: ManagedSaveAttempt,
    tmp_path: Path,
) -> SaveAttemptExecutionContext:
    run_id = "run-1"
    difficulty = attempt.difficulty or "expert"
    cup_id = attempt.cup_id or "jack"
    target = ManagedSaveUnlockTarget(
        sequence_index=0,
        kind=attempt.target_kind or "clear_gp_cup",
        status="pending",
        label=f"{difficulty.title()} {cup_id.title()} Cup",
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=attempt.course_id,
    )
    course_setup_target = CourseSetupTarget(
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=attempt.course_id,
    )
    course_setup = ManagedSaveCourseSetup(
        id="course-setup-1",
        save_game_id=save_game.id,
        policy_run_id=run_id,
        policy_artifact="latest",
        engine_setting_raw_value=50,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=attempt.course_id,
    )
    cup_setup = ManagedSaveCupSetup(
        id="cup-setup-1",
        save_game_id=save_game.id,
        cup_id=cup_id,
        vehicle_id="blue_falcon",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        difficulty=difficulty,
    )
    policy_run = ManagedRun(
        id=run_id,
        name="Run",
        status="finished",
        config=default_managed_run_config(),
        config_hash="hash",
        run_dir=tmp_path / "run-1",
        created_at="2026-01-01T00:00:00Z",
        lineage_id="lineage-1",
    )
    return SaveAttemptExecutionContext(
        save_game=save_game,
        attempt=attempt,
        target=target,
        course_setup_target=course_setup_target,
        course_setup=course_setup,
        cup_setup=cup_setup,
        policy_run=policy_run,
        policy_artifact="latest",
        policy_path=policy_run.run_dir / "latest.zip",
    )
