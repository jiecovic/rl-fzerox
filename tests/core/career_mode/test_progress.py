# tests/core/career_mode/test_progress.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.core.career_mode.progress import (
    build_unlock_progress,
    default_unlock_targets,
)
from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.runner.progress import CareerAttemptProgress
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
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
