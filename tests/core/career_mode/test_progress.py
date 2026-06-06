# tests/core/career_mode/test_progress.py

from pathlib import Path

from rl_fzerox.core.career_mode.progress import (
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


def test_build_unlock_progress_returns_pending_targets_when_save_is_missing() -> None:
    progress = build_unlock_progress(Path("/tmp/fzerox.srm"))

    assert progress.inspection_status == "not_inspected"
    assert progress.completed_count == 0
    assert progress.total_count == len(default_unlock_targets())
    assert progress.next_target is not None
    assert progress.next_target.difficulty == "novice"
    assert progress.next_target.cup_id == "jack"


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
