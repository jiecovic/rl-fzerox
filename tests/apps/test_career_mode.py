# tests/apps/test_career_mode.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.apps.career_mode_cli.config import (
    career_mode_base_config,
    career_policy_observation_layout_shape_hint,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig, EmulatorConfig


def test_career_mode_base_config_uses_save_runtime_without_policy_template(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    save_path = tmp_path / "save" / "fzerox.srm"
    core_path.touch()
    rom_path.touch()
    save_path.parent.mkdir()
    save_path.write_bytes(b"save")

    config = career_mode_base_config(
        db_path=tmp_path / "manager.db",
        save_game_id="save-a",
        save_path=save_path,
        attempt_id="attempt-a",
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        attempt_seed=1234,
        deterministic_policy=False,
        race_setup=CareerModeRaceSetupConfig(
            difficulty="novice",
            cup_id="jack",
            vehicle_id="blue_falcon",
            vehicle_display_name="Blue Falcon",
            character_index=1,
            machine_select_slot=1,
            machine_select_row=0,
            machine_select_column=1,
            engine_setting_raw_value=50,
        ),
        label="Clear Novice Jack Cup",
    )

    assert config.seed is None
    assert config.emulator.core_path == core_path
    assert config.emulator.rom_path == rom_path
    assert config.emulator.baseline_state_path is None
    assert config.emulator.runtime_dir == save_path.parent / "runtime"
    assert config.policy is None
    assert config.train is None
    assert config.watch.manager_db_path == tmp_path / "manager.db"
    assert config.watch.managed_save_game_id == "save-a"
    assert config.watch.save_attempt_id == "attempt-a"
    assert config.watch.unlock_target_label == "Clear Novice Jack Cup"
    assert config.watch.policy_run_dir is None
    assert config.watch.policy_artifact == "latest"
    assert config.watch.policy_algorithm is None
    assert config.watch.policy_observation_layout_shape_hint is None
    assert config.watch.attempt_seed == 1234
    assert config.watch.deterministic_policy is False
    assert config.watch.control_fps == "auto"
    assert config.watch.start_manual_control is False
    assert config.watch.career_mode_race_setup is not None
    assert config.watch.career_mode_race_setup.cup_id == "jack"


def test_career_policy_observation_layout_shape_hint_reserves_largest_preview() -> None:
    wide_stack = ManagedRunConfig.model_validate(
        {
            "observation": {
                "resolution": {"mode": "custom", "height": 72, "width": 120},
                "frame_stack": 4,
            }
        }
    )
    tall_minimap = ManagedRunConfig.model_validate(
        {
            "observation": {
                "resolution": {"mode": "custom", "height": 100, "width": 80},
                "frame_stack": 2,
                "minimap_layer": True,
            }
        }
    )

    assert career_policy_observation_layout_shape_hint((wide_stack, tall_minimap)) == (
        100,
        120,
        12,
    )
