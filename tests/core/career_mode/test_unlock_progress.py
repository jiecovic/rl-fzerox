# tests/core/career_mode/test_unlock_progress.py
"""Tests for rule-derived Career Mode save unlock progress."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.career_mode.progress.unlocks import (
    build_unlock_progress,
    default_unlock_targets,
)
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


def _logical_sra(cup_progress: dict[str, int]) -> bytes:
    payload = bytearray(FZEROX_SAVE_LAYOUT.raw_sra_size)
    payload[: len(FZEROX_SAVE_LAYOUT.title)] = FZEROX_SAVE_LAYOUT.title
    for progress_offset in FZEROX_SAVE_LAYOUT.gp_progress_offsets:
        payload[progress_offset.offset] = cup_progress.get(progress_offset.cup_id, 0)
    return bytes(payload)
