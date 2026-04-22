//! Native-owned RAM patching for deterministic F-Zero X race-start baselines.

use crate::core::error::CoreError;
use crate::core::game::memory::{
    read_f32, read_i8, read_i16, read_i32, write_f32, write_i8, write_i16, write_i32,
};
use crate::core::game::telemetry::layout::{
    GLOBALS, GameMode, MACHINE_TABLE, RACER, TELEMETRY_CONFIG,
};

#[derive(Clone, Copy, Debug)]
pub struct RaceStartSetup {
    pub course_index: i32,
    pub character_index: i16,
    pub machine_skin_index: i16,
    pub engine_setting_raw_value: i32,
    pub total_lap_count: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct VehicleSetupInfo {
    pub character_index: i16,
    pub engine_setting: f32,
    pub character_engine_setting: f32,
    pub racer_engine_curve: f32,
}

#[derive(Clone, Copy)]
struct RaceStartGameIds {
    time_attack_menu_mode: i32,
    no_ghost: i32,
    single_player: i32,
    change_init: i16,
}

#[derive(Clone, Copy)]
struct RaceStartBounds {
    course_count: i32,
    engine_setting_min: i32,
    engine_setting_max: i32,
    minimum_lap_count: i32,
}

const GAME_IDS: RaceStartGameIds = RaceStartGameIds {
    time_attack_menu_mode: 1,
    no_ghost: 0,
    single_player: 1,
    change_init: 3,
};

const BOUNDS: RaceStartBounds = RaceStartBounds {
    course_count: 24,
    engine_setting_min: 0,
    engine_setting_max: 100,
    minimum_lap_count: 1,
};

pub fn write_time_attack_race_setup(
    system_ram: &mut [u8],
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(setup)?;
    write_menu_setup(system_ram, setup)?;
    write_live_racer_setup(system_ram, setup)
}

pub fn write_time_attack_machine_settings(
    system_ram: &mut [u8],
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(setup)?;
    write_menu_setup(system_ram, setup)
}

pub fn force_time_attack_reinit(system_ram: &mut [u8]) -> Result<(), CoreError> {
    write_i32(system_ram, GLOBALS.game_mode, GameMode::TimeAttack as i32)?;
    write_i32(
        system_ram,
        GLOBALS.queued_game_mode,
        GameMode::TimeAttack as i32,
    )?;
    write_i16(
        system_ram,
        GLOBALS.game_mode_change_state,
        GAME_IDS.change_init,
    )
}

pub fn validate_time_attack_race_setup(
    system_ram: &[u8],
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(setup)?;
    let engine_value = setup.engine_setting_raw_value as f32 / 100.0;
    let engine_curve = engine_to_curve_value(engine_value);
    let player_base = player_racer_base();

    let mut mismatches = Vec::new();
    push_mismatch(
        &mut mismatches,
        "course_index",
        read_i32(system_ram, GLOBALS.course_index)?,
        setup.course_index,
    );
    push_mismatch(
        &mut mismatches,
        "selected_mode",
        read_i32(system_ram, GLOBALS.selected_mode)?,
        GAME_IDS.time_attack_menu_mode,
    );
    push_mismatch(
        &mut mismatches,
        "current_ghost_type",
        read_i32(system_ram, GLOBALS.current_ghost_type)?,
        GAME_IDS.no_ghost,
    );
    push_mismatch(
        &mut mismatches,
        "player_character",
        read_i16(system_ram, GLOBALS.player_characters)?,
        setup.character_index,
    );
    push_mismatch(
        &mut mismatches,
        "racer_character",
        read_i8(system_ram, player_base + RACER.character)?,
        setup.character_index as i8,
    );
    push_mismatch(
        &mut mismatches,
        "racer_machine_skin",
        read_i16(system_ram, player_base + RACER.machine_skin_index)?,
        setup.machine_skin_index,
    );

    let player_engine = read_f32(system_ram, GLOBALS.player_engine)?;
    if (player_engine - engine_value).abs() > 0.001 {
        mismatches.push(format!(
            "player_engine: expected {engine_value:.3}, got {player_engine:.3}"
        ));
    }
    let racer_engine_curve = read_f32(system_ram, player_base + RACER.engine_curve)?;
    if (racer_engine_curve - engine_curve).abs() > 0.001 {
        mismatches.push(format!(
            "racer_engine_curve: expected {engine_curve:.3}, got {racer_engine_curve:.3}"
        ));
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(CoreError::InvalidRaceStartSetup {
            message: mismatches.join("; "),
        })
    }
}

pub fn vehicle_setup_info(system_ram: &[u8]) -> Result<VehicleSetupInfo, CoreError> {
    let character_index = read_i16(system_ram, GLOBALS.player_characters)?;
    let player_base = player_racer_base();
    Ok(VehicleSetupInfo {
        character_index,
        engine_setting: read_f32(system_ram, GLOBALS.player_engine)?,
        character_engine_setting: read_f32(
            system_ram,
            GLOBALS.character_last_engine + ((character_index as usize) * 4),
        )?,
        racer_engine_curve: read_f32(system_ram, player_base + RACER.engine_curve)?,
    })
}

fn write_menu_setup(system_ram: &mut [u8], setup: RaceStartSetup) -> Result<(), CoreError> {
    let engine_value = setup.engine_setting_raw_value as f32 / 100.0;

    write_i32(system_ram, GLOBALS.num_players, GAME_IDS.single_player)?;
    write_i32(system_ram, GLOBALS.total_lap_count, setup.total_lap_count)?;
    write_i32(
        system_ram,
        GLOBALS.selected_mode,
        GAME_IDS.time_attack_menu_mode,
    )?;
    write_i32(system_ram, GLOBALS.current_ghost_type, GAME_IDS.no_ghost)?;
    write_i32(system_ram, GLOBALS.course_index, setup.course_index)?;
    write_i16(system_ram, GLOBALS.player_characters, setup.character_index)?;
    write_i16(
        system_ram,
        GLOBALS.player_machine_skins,
        setup.machine_skin_index,
    )?;
    write_f32(system_ram, GLOBALS.player_engine, engine_value)?;
    write_f32(
        system_ram,
        GLOBALS.character_last_engine + ((setup.character_index as usize) * 4),
        engine_value,
    )
}

fn write_live_racer_setup(system_ram: &mut [u8], setup: RaceStartSetup) -> Result<(), CoreError> {
    let engine_value = setup.engine_setting_raw_value as f32 / 100.0;
    let player_base = player_racer_base();
    write_i8(
        system_ram,
        player_base + RACER.character,
        setup.character_index as i8,
    )?;
    write_i16(
        system_ram,
        player_base + RACER.machine_skin_index,
        setup.machine_skin_index,
    )?;
    write_f32(
        system_ram,
        player_base + RACER.engine_curve,
        engine_to_curve_value(engine_value),
    )
}

fn validate_setup(setup: RaceStartSetup) -> Result<(), CoreError> {
    if !(0..BOUNDS.course_count).contains(&setup.course_index) {
        return Err(invalid_setup(format!(
            "course_index must be in [0, {}), got {}",
            BOUNDS.course_count, setup.course_index
        )));
    }
    if setup.character_index < 0 || setup.character_index as usize >= MACHINE_TABLE.machine_count {
        return Err(invalid_setup(format!(
            "character_index must be in [0, {}), got {}",
            MACHINE_TABLE.machine_count, setup.character_index
        )));
    }
    if setup.machine_skin_index < 0 {
        return Err(invalid_setup(format!(
            "machine_skin_index must be non-negative, got {}",
            setup.machine_skin_index
        )));
    }
    if !(BOUNDS.engine_setting_min..=BOUNDS.engine_setting_max)
        .contains(&setup.engine_setting_raw_value)
    {
        return Err(invalid_setup(format!(
            "engine_setting raw value must be in [{}, {}], got {}",
            BOUNDS.engine_setting_min, BOUNDS.engine_setting_max, setup.engine_setting_raw_value
        )));
    }
    if setup.total_lap_count < BOUNDS.minimum_lap_count {
        return Err(invalid_setup(format!(
            "total_lap_count must be at least {}, got {}",
            BOUNDS.minimum_lap_count, setup.total_lap_count
        )));
    }
    Ok(())
}

fn invalid_setup(message: String) -> CoreError {
    CoreError::InvalidRaceStartSetup { message }
}

fn player_racer_base() -> usize {
    GLOBALS.racers + (TELEMETRY_CONFIG.player_racer_index * RACER.size)
}

fn engine_to_curve_value(engine_value: f32) -> f32 {
    if engine_value == 0.0 {
        0.0
    } else {
        1.0 / (((1.0 + 0.689_999_8) / engine_value) - 0.689_999_8)
    }
}

fn push_mismatch<T>(mismatches: &mut Vec<String>, label: &str, actual: T, expected: T)
where
    T: std::fmt::Debug + PartialEq,
{
    if actual != expected {
        mismatches.push(format!("{label}: expected {expected:?}, got {actual:?}"));
    }
}
