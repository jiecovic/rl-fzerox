# tests/ui/test_career_mode_debug.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig
from rl_fzerox.ui.watch.runtime.career_mode.loop.debug import CareerModeDebugTrace


def test_career_mode_debug_trace_writes_rank_and_frame(tmp_path: Path) -> None:
    db_path = tmp_path / "manager.db"
    store = ManagerStore(db_path)
    save_game = store.create_save_game(name="Career Save", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="master",
        cup_id="joker",
    )
    controller = CareerModeController(
        _race_setup(),
        db_path=db_path,
        save_game_id=save_game.id,
        attempt_id=attempt.id,
        device="cpu",
        single_target=True,
        target_clear_goal=1,
    )
    trace = CareerModeDebugTrace(tmp_path / "career.debug")
    frame: RgbFrame = np.zeros((2, 3, 3), dtype=np.uint8)
    info = controller.viewer_info(
        info={
            "game_mode": "gp_end_cutscene",
            "game_mode_raw": 13,
            "course_index": 23,
            "position": 1,
            "gp_final_rank": 1,
            "termination_reason": "finished",
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
        active_policy_control=None,
    )

    trace.observe(
        stage="post_gp",
        event="recording_close:succeeded",
        info=info,
        controller=controller,
        frame_source=lambda: frame,
        force=True,
    )

    entries = [
        json.loads(line)
        for line in (tmp_path / "career.debug" / "trace.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["event"] == "recording_close:succeeded"
    assert entries[0]["values"]["game_mode"] == "gp_end_cutscene"
    assert entries[0]["values"]["gp_final_rank"] == 1
    assert entries[0]["values"]["position"] == 1

    screenshot = tmp_path / "career.debug" / entries[0]["screenshot"]
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_career_mode_debug_trace_is_sparse_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "manager.db"
    store = ManagerStore(db_path)
    save_game = store.create_save_game(name="Career Save", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    controller = CareerModeController(
        _race_setup(),
        db_path=db_path,
        save_game_id=save_game.id,
        attempt_id=attempt.id,
        device="cpu",
    )
    trace = CareerModeDebugTrace(tmp_path / "career.debug", screenshots=False)
    info = controller.viewer_info(
        info={"game_mode": "main_menu", "game_mode_raw": 2},
        active_policy_control=None,
    )

    trace.observe(stage="loop", info=info, controller=controller)
    trace.observe(stage="loop", info=info, controller=controller)

    lines = (tmp_path / "career.debug" / "trace.jsonl").read_text().splitlines()
    assert len(lines) == 1


def _race_setup() -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="master",
        cup_id="joker",
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=1,
        machine_select_slot=1,
        machine_select_row=0,
        machine_select_column=1,
        engine_setting_raw_value=103,
    )
