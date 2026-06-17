# src/rl_fzerox/core/career_mode/execution/setup.py
from rl_fzerox.core.career_mode.execution.race import SaveRaceSetup
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig, WatchAppConfig


def save_race_setup_from_config(config: WatchAppConfig) -> CareerModeRaceSetupConfig:
    setup = config.watch.career_mode_race_setup
    if setup is None:
        raise RuntimeError("career mode requires watch.career_mode_race_setup")
    return setup


def career_mode_race_setup_config(
    race_setup: SaveRaceSetup,
) -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty=race_setup.difficulty,
        cup_id=race_setup.cup_id,
        course_id=race_setup.course_id,
        vehicle_id=race_setup.vehicle_id,
        vehicle_display_name=race_setup.vehicle_display_name,
        character_index=race_setup.character_index,
        machine_select_slot=race_setup.machine_select_slot,
        machine_select_row=race_setup.machine_select_row,
        machine_select_column=race_setup.machine_select_column,
        engine_setting_raw_value=race_setup.engine_setting_raw_value,
    )
