# tests/core/career_mode/test_progress.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.career_mode.course_setup import CourseSetupTarget
from rl_fzerox.core.career_mode.progress import (
    build_unlock_progress,
    default_unlock_targets,
)
from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.runner.progress import CareerAttemptProgress
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
from rl_fzerox.core.save_game.unlocks import FZEROX_SAVE_LAYOUT


def test_default_unlock_targets_cover_fixed_gp_cups_by_difficulty() -> None:
    targets = default_unlock_targets()

    assert len(targets) == 16
    assert targets[0].kind == "clear_gp_cup"
    assert targets[0].difficulty == "novice"
    assert targets[0].cup_id == "jack"
    assert targets[-1].difficulty == "master"
    assert targets[-1].cup_id == "joker"


def test_default_unlock_targets_do_not_include_x_cup() -> None:
    targets = default_unlock_targets()

    assert all(target.cup_id != "x" for target in targets)


def test_build_unlock_progress_only_starts_first_target_when_save_is_missing() -> None:
    progress = build_unlock_progress(Path("/tmp/fzerox.srm"))

    assert progress.inspection_status == "not_inspected"
    assert progress.completed_count == 0
    assert progress.total_count == len(default_unlock_targets())
    assert progress.unlocked_vehicle_count == 6
    assert progress.unlocked_vehicle_ids[:2] == ("blue_falcon", "golden_fox")
    assert progress.next_target is not None
    assert progress.next_target.difficulty == "novice"
    assert progress.next_target.cup_id == "jack"
    assert progress.targets[0].status == "pending"
    assert {target.status for target in progress.targets[1:]} == {"locked"}


def test_build_unlock_progress_marks_completed_gp_cups_from_save(tmp_path: Path) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 2, "queen": 1}))

    progress = build_unlock_progress(save_path)

    assert progress.inspection_status == "inspected"
    assert progress.completed_count == 3
    assert progress.total_count == len(default_unlock_targets())
    assert progress.next_target is not None
    assert progress.next_target.difficulty == "novice"
    assert progress.next_target.cup_id == "king"


def test_build_unlock_progress_skips_joker_until_standard_initial_cups_are_clear(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 1, "queen": 1, "king": 1}))

    progress = build_unlock_progress(save_path)

    novice_joker = next(
        target
        for target in progress.targets
        if target.difficulty == "novice" and target.cup_id == "joker"
    )
    assert novice_joker.status == "locked"
    assert progress.next_target is not None
    assert progress.next_target.difficulty == "standard"
    assert progress.next_target.cup_id == "jack"
    assert progress.unlocked_vehicle_count == 12
    assert progress.unlocked_vehicle_ids[-1] == "mad_wolf"
    assert "mighty_hurricane" not in progress.unlocked_vehicle_ids


def test_build_unlock_progress_unlocks_joker_after_standard_initial_cups_are_clear(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 2, "queen": 2, "king": 2}))

    progress = build_unlock_progress(save_path)

    assert progress.next_target is not None
    assert progress.next_target.difficulty == "novice"
    assert progress.next_target.cup_id == "joker"


def test_build_unlock_progress_unlocks_master_after_all_expert_cups_are_clear(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 3, "queen": 3, "king": 3, "joker": 3}))

    progress = build_unlock_progress(save_path)

    master_targets = tuple(target for target in progress.targets if target.difficulty == "master")
    assert {target.status for target in master_targets} == {"pending"}
    assert progress.next_target is not None
    assert progress.next_target.difficulty == "master"
    assert progress.next_target.cup_id == "jack"


def test_build_unlock_progress_locks_master_until_all_expert_cups_are_clear(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 2, "queen": 3, "king": 3, "joker": 3}))

    progress = build_unlock_progress(save_path)

    master_targets = tuple(target for target in progress.targets if target.difficulty == "master")
    assert {target.status for target in master_targets} == {"locked"}


def test_build_unlock_progress_finishes_when_all_targets_are_clear(tmp_path: Path) -> None:
    save_path = tmp_path / "fzerox.sra"
    save_path.write_bytes(_logical_sra({"jack": 4, "queen": 4, "king": 4, "joker": 4}))

    progress = build_unlock_progress(save_path)

    assert progress.inspection_status == "inspected"
    assert progress.completed_count == progress.total_count == 16
    assert progress.next_target is None


def test_crashed_race_keeps_current_gp_attempt(tmp_path: Path) -> None:
    store = _Store(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "crashed", "position": 30, "race_time_ms": 88_333},
    )

    assert transition.attempt_finished is False
    assert transition.next_plan is None
    assert store.finished_attempts == []
    assert store.started_next_attempt_count == 0
    assert store.save_game.save_path.read_bytes() == b"save-ram"


def test_single_target_success_waits_for_post_gp_screen(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    result_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "results", "termination_reason": "finished"},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert result_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is True
    assert post_gp_transition.next_plan is None
    assert post_gp_transition.finished_attempt_id == "attempt-1"
    assert post_gp_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_success_finishes_on_unskippable_credits(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.next_plan is None
    assert credits_transition.finished_attempt_id == "attempt-1"
    assert credits_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_success_finishes_after_return_to_main_menu(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.next_plan is None
    assert menu_transition.finished_attempt_id == "attempt-1"
    assert menu_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_success_does_not_finish_on_mid_cup_machine_settings(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "machine_settings", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0


def test_replayed_target_retire_does_not_finish_attempt(tmp_path: Path) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "retired", "position": 30, "race_time_ms": 12_345},
    )

    assert transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0


def test_single_target_failed_gp_exit_starts_retry_attempt(tmp_path: Path) -> None:
    store = _FailedRetryStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "retired", "position": 30, "race_time_ms": 12_345},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_attempt_id == "attempt-1"
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp attempt returned to menu before cup clear"
    assert menu_transition.next_plan is not None
    assert menu_transition.next_plan.attempt_id == "attempt-2"
    assert progress.attempt_id == "attempt-1"
    progress.apply_execution_plan(menu_transition.next_plan)
    assert progress.attempt_id == "attempt-2"
    assert store.finished_attempts == [
        ("attempt-1", "failed", "gp attempt returned to menu before cup clear")
    ]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_replayed_target_success_waits_for_post_gp_screen(tmp_path: Path) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 5,
        },
    )
    result_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "results", "termination_reason": "finished"},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert result_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is True
    assert post_gp_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_replayed_target_success_does_not_finish_after_nonfinal_course(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 0,
        },
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0


def _logical_sra(cup_progress: dict[str, int]) -> bytes:
    payload = bytearray(FZEROX_SAVE_LAYOUT.raw_sra_size)
    payload[: len(FZEROX_SAVE_LAYOUT.title)] = FZEROX_SAVE_LAYOUT.title
    for progress_offset in FZEROX_SAVE_LAYOUT.gp_progress_offsets:
        payload[progress_offset.offset] = cup_progress.get(progress_offset.cup_id, 0)
    return bytes(payload)


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

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        raise AssertionError("no execution context should be requested")


class _SingleTargetCompletionStore(_Store):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.status_updates: list[SaveGameStatus] = []
        self.progress_reads = 0

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

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        raise AssertionError("single-target success must not request the next context")


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

    def get_save_attempt_execution_context(
        self,
        attempt_id: str,
    ) -> SaveAttemptExecutionContext | None:
        if self.next_attempt is None or attempt_id != self.next_attempt.id:
            return None
        return SaveAttemptExecutionContext(
            save_game=self.save_game,
            attempt=self.next_attempt,
            target=ManagedSaveUnlockTarget(
                sequence_index=0,
                kind="clear_gp_cup",
                status="pending",
                label="Expert Jack Cup",
                difficulty="expert",
                cup_id="jack",
            ),
            course_setup_target=CourseSetupTarget(
                difficulty="expert",
                cup_id="jack",
                course_id=None,
            ),
            course_setup=ManagedSaveCourseSetup(
                id="course-setup",
                save_game_id=self.save_game.id,
                policy_run_id="policy-run",
                policy_artifact="latest",
                engine_setting_raw_value=50,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                difficulty="expert",
                cup_id="jack",
                course_id=None,
            ),
            cup_setup=ManagedSaveCupSetup(
                id="cup-setup",
                save_game_id=self.save_game.id,
                cup_id="jack",
                vehicle_id="blue_falcon",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                difficulty="expert",
            ),
            policy_run=ManagedRun(
                id="policy-run",
                name="Policy Run",
                status="finished",
                config=default_managed_run_config(),
                config_hash="policy-run-hash",
                run_dir=self.tmp_path / "policy-run",
                created_at="2026-01-01T00:00:00Z",
                lineage_id="policy-run",
            ),
            policy_artifact="latest",
            policy_path=self.tmp_path / "policy-run" / "latest.zip",
        )


def _race_setup() -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="expert",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_raw_value=50,
    )
