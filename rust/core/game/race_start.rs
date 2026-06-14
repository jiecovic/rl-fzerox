// rust/core/game/race_start.rs
//! Native-owned RAM patching for deterministic F-Zero X race-start baselines.

use crate::core::error::CoreError;
use crate::core::game::memory::{
    read_f32, read_i8, read_i16, read_i32, write_f32, write_i8, write_i16, write_i32,
};
use crate::core::game::race_start::engine_physics::{
    engine_to_curve_value, write_live_engine_fields,
};
use crate::core::game::telemetry::layout::{
    GLOBALS, GameMode, MACHINE_TABLE, RACER, TELEMETRY_CONFIG,
};

mod engine_physics;

#[derive(Clone, Copy, Debug)]
pub struct RaceStartSetup {
    pub course_index: i32,
    pub character_index: i16,
    pub machine_skin_index: i16,
    pub engine_setting_raw_value: i32,
    pub total_lap_count: i32,
    pub gp_difficulty_raw_value: i32,
}

#[derive(Clone, Copy)]
struct RaceStartSentinels {
    preserve_machine_skin: i16,
    preserve_gp_difficulty: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct VehicleSetupInfo {
    pub player_character_index: i16,
    pub racer_character_index: i16,
    pub engine_setting: f32,
    pub character_engine_setting: Option<f32>,
    pub racer_engine_curve: Option<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RaceStartMode {
    TimeAttack,
    GpRace,
}

#[derive(Clone, Copy)]
struct RaceStartMenuState {
    selected_mode: Option<i32>,
    current_ghost_type: Option<i32>,
}

#[derive(Clone, Copy)]
struct RaceStartGameIds {
    single_player: i32,
    change_init: i16,
}

#[derive(Clone, Copy)]
struct RaceStartBounds {
    built_in_course_count: i32,
    x_cup_course_index: i32,
    engine_setting_min: i32,
    engine_setting_max: i32,
    minimum_lap_count: i32,
}

const GAME_IDS: RaceStartGameIds = RaceStartGameIds {
    single_player: 1,
    change_init: 3,
};

const BOUNDS: RaceStartBounds = RaceStartBounds {
    built_in_course_count: 24,
    x_cup_course_index: 48,
    engine_setting_min: 0,
    engine_setting_max: 100,
    minimum_lap_count: 1,
};

const SENTINELS: RaceStartSentinels = RaceStartSentinels {
    preserve_machine_skin: -1,
    preserve_gp_difficulty: -1,
};

pub fn write_time_attack_menu_mode(system_ram: &mut [u8]) -> Result<(), CoreError> {
    write_menu_mode_state(system_ram, RaceStartMode::TimeAttack)
}

pub fn vehicle_setup_info(system_ram: &[u8]) -> Result<VehicleSetupInfo, CoreError> {
    let player_character_index = read_i16(system_ram, GLOBALS.player_characters)?;
    let racer_character_index = read_i8(system_ram, player_racer_base() + RACER.character)? as i16;
    let player_base = player_racer_base();
    let active_character_index = active_character_index(system_ram)?;
    Ok(VehicleSetupInfo {
        player_character_index,
        racer_character_index,
        engine_setting: read_f32(system_ram, GLOBALS.player_engine)?,
        character_engine_setting: valid_character_index(active_character_index)
            .then(|| {
                read_f32(
                    system_ram,
                    GLOBALS.character_last_engine + ((active_character_index as usize) * 4),
                )
            })
            .transpose()?,
        racer_engine_curve: valid_character_index(racer_character_index)
            .then(|| read_f32(system_ram, player_base + RACER.engine_curve))
            .transpose()?,
    })
}

pub fn write_race_setup(
    system_ram: &mut [u8],
    mode: RaceStartMode,
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(mode, setup)?;
    write_menu_setup(system_ram, mode, setup)?;
    write_live_racer_setup(system_ram, setup)
}

pub fn write_machine_settings(
    system_ram: &mut [u8],
    mode: RaceStartMode,
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(mode, setup)?;
    write_menu_setup(system_ram, mode, setup)
}

pub fn write_engine_settings(
    system_ram: &mut [u8],
    _mode: RaceStartMode,
    engine_setting_raw_value: i32,
) -> Result<(), CoreError> {
    write_engine_settings_only(system_ram, engine_setting_raw_value)
}

fn write_menu_mode_state(system_ram: &mut [u8], mode: RaceStartMode) -> Result<(), CoreError> {
    write_i32(system_ram, GLOBALS.num_players, GAME_IDS.single_player)?;
    if let Some(selected_mode) = mode.menu_state().selected_mode {
        write_i32(system_ram, GLOBALS.selected_mode, selected_mode)?;
    }
    if let Some(current_ghost_type) = mode.menu_state().current_ghost_type {
        write_i32(system_ram, GLOBALS.current_ghost_type, current_ghost_type)?;
    }
    Ok(())
}

fn write_engine_settings_only(
    system_ram: &mut [u8],
    engine_setting_raw_value: i32,
) -> Result<(), CoreError> {
    let character_index = active_character_index(system_ram)?;
    validate_character_and_engine(character_index, engine_setting_raw_value)?;
    let engine_value = engine_setting_raw_value as f32 / 100.0;
    let player_base = player_racer_base();

    write_f32(system_ram, GLOBALS.player_engine, engine_value)?;
    write_f32(
        system_ram,
        GLOBALS.character_last_engine + ((character_index as usize) * 4),
        engine_value,
    )?;
    write_live_engine_fields(system_ram, character_index, engine_value, player_base)
}

pub fn force_race_reinit(system_ram: &mut [u8], mode: RaceStartMode) -> Result<(), CoreError> {
    let game_mode = mode.game_mode() as i32;
    write_i32(system_ram, GLOBALS.game_mode, game_mode)?;
    write_i32(system_ram, GLOBALS.queued_game_mode, game_mode)?;
    write_i16(
        system_ram,
        GLOBALS.game_mode_change_state,
        GAME_IDS.change_init,
    )
}

pub fn validate_race_setup(
    system_ram: &[u8],
    mode: RaceStartMode,
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    validate_setup(mode, setup)?;
    let engine_value = setup.engine_setting_raw_value as f32 / 100.0;
    let engine_curve = engine_to_curve_value(engine_value);
    let player_base = player_racer_base();

    let player_character = read_i16(system_ram, GLOBALS.player_characters)?;
    let racer_character = read_i8(system_ram, player_base + RACER.character)? as i16;
    let mut mismatches = Vec::new();
    push_mismatch(
        &mut mismatches,
        "course_index",
        read_i32(system_ram, GLOBALS.course_index)?,
        setup.course_index,
    );
    if let Some(selected_mode) = mode.menu_state().selected_mode {
        push_mismatch(
            &mut mismatches,
            "selected_mode",
            read_i32(system_ram, GLOBALS.selected_mode)?,
            selected_mode,
        );
    }
    if let Some(current_ghost_type) = mode.menu_state().current_ghost_type {
        push_mismatch(
            &mut mismatches,
            "current_ghost_type",
            read_i32(system_ram, GLOBALS.current_ghost_type)?,
            current_ghost_type,
        );
    }
    if mode == RaceStartMode::GpRace
        && setup.gp_difficulty_raw_value != SENTINELS.preserve_gp_difficulty
    {
        push_mismatch(
            &mut mismatches,
            "difficulty",
            read_i32(system_ram, GLOBALS.difficulty)?,
            setup.gp_difficulty_raw_value,
        );
    }
    if player_character != setup.character_index && racer_character != setup.character_index {
        mismatches.push(format!(
            "character: expected {:?}, got player_character={:?}, racer_character={:?}",
            setup.character_index, player_character, racer_character
        ));
    }
    if setup.machine_skin_index != SENTINELS.preserve_machine_skin {
        push_mismatch(
            &mut mismatches,
            "racer_machine_skin",
            read_i16(system_ram, player_base + RACER.machine_skin_index)?,
            setup.machine_skin_index,
        );
    }

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

fn write_menu_setup(
    system_ram: &mut [u8],
    mode: RaceStartMode,
    setup: RaceStartSetup,
) -> Result<(), CoreError> {
    let engine_value = setup.engine_setting_raw_value as f32 / 100.0;

    write_i32(system_ram, GLOBALS.num_players, GAME_IDS.single_player)?;
    write_i32(system_ram, GLOBALS.total_lap_count, setup.total_lap_count)?;
    if let Some(selected_mode) = mode.menu_state().selected_mode {
        write_i32(system_ram, GLOBALS.selected_mode, selected_mode)?;
    }
    if let Some(current_ghost_type) = mode.menu_state().current_ghost_type {
        write_i32(system_ram, GLOBALS.current_ghost_type, current_ghost_type)?;
    }
    if mode == RaceStartMode::GpRace
        && setup.gp_difficulty_raw_value != SENTINELS.preserve_gp_difficulty
    {
        write_i32(
            system_ram,
            GLOBALS.difficulty,
            setup.gp_difficulty_raw_value,
        )?;
    }
    write_i32(system_ram, GLOBALS.course_index, setup.course_index)?;
    write_i16(system_ram, GLOBALS.player_characters, setup.character_index)?;
    if setup.machine_skin_index != SENTINELS.preserve_machine_skin {
        write_i16(
            system_ram,
            GLOBALS.player_machine_skins,
            setup.machine_skin_index,
        )?;
    }
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
    if setup.machine_skin_index != SENTINELS.preserve_machine_skin {
        write_i16(
            system_ram,
            player_base + RACER.machine_skin_index,
            setup.machine_skin_index,
        )?;
    }
    write_live_engine_fields(system_ram, setup.character_index, engine_value, player_base)
}

fn validate_setup(mode: RaceStartMode, setup: RaceStartSetup) -> Result<(), CoreError> {
    validate_course_index(mode, setup.course_index)?;
    validate_character_and_engine(setup.character_index, setup.engine_setting_raw_value)?;
    if setup.machine_skin_index < 0 && setup.machine_skin_index != SENTINELS.preserve_machine_skin {
        return Err(invalid_setup(format!(
            "machine_skin_index must be non-negative or preserve sentinel {}, got {}",
            SENTINELS.preserve_machine_skin, setup.machine_skin_index
        )));
    }
    if setup.total_lap_count < BOUNDS.minimum_lap_count {
        return Err(invalid_setup(format!(
            "total_lap_count must be at least {}, got {}",
            BOUNDS.minimum_lap_count, setup.total_lap_count
        )));
    }
    if mode == RaceStartMode::GpRace
        && setup.gp_difficulty_raw_value != SENTINELS.preserve_gp_difficulty
        && !(0..=3).contains(&setup.gp_difficulty_raw_value)
    {
        return Err(invalid_setup(format!(
            "gp_difficulty_raw_value must be in [0, 3] or preserve sentinel {}, got {}",
            SENTINELS.preserve_gp_difficulty, setup.gp_difficulty_raw_value
        )));
    }
    Ok(())
}

fn validate_course_index(mode: RaceStartMode, course_index: i32) -> Result<(), CoreError> {
    if (0..BOUNDS.built_in_course_count).contains(&course_index) {
        return Ok(());
    }
    if mode == RaceStartMode::GpRace && course_index == BOUNDS.x_cup_course_index {
        return Ok(());
    }

    let supported_range = match mode {
        RaceStartMode::TimeAttack => format!("[0, {})", BOUNDS.built_in_course_count),
        RaceStartMode::GpRace => format!(
            "[0, {}) or {} for X Cup",
            BOUNDS.built_in_course_count, BOUNDS.x_cup_course_index
        ),
    };
    Err(invalid_setup(format!(
        "course_index must be {supported_range}, got {course_index}"
    )))
}

fn validate_character_and_engine(
    character_index: i16,
    engine_setting_raw_value: i32,
) -> Result<(), CoreError> {
    if character_index < 0 || character_index as usize >= MACHINE_TABLE.machine_count {
        return Err(invalid_setup(format!(
            "character_index must be in [0, {}), got {}",
            MACHINE_TABLE.machine_count, character_index
        )));
    }
    if !(BOUNDS.engine_setting_min..=BOUNDS.engine_setting_max).contains(&engine_setting_raw_value)
    {
        return Err(invalid_setup(format!(
            "engine_setting raw value must be in [{}, {}], got {}",
            BOUNDS.engine_setting_min, BOUNDS.engine_setting_max, engine_setting_raw_value
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

fn active_character_index(system_ram: &[u8]) -> Result<i16, CoreError> {
    let racer_character_index = read_i8(system_ram, player_racer_base() + RACER.character)? as i16;
    if valid_character_index(racer_character_index) {
        return Ok(racer_character_index);
    }
    read_i16(system_ram, GLOBALS.player_characters)
}

fn valid_character_index(character_index: i16) -> bool {
    character_index >= 0 && (character_index as usize) < MACHINE_TABLE.machine_count
}

impl RaceStartMode {
    pub fn parse(raw: &str) -> Result<Self, CoreError> {
        match raw {
            "time_attack" => Ok(Self::TimeAttack),
            "gp_race" => Ok(Self::GpRace),
            _ => Err(invalid_setup(format!(
                "unsupported race-start mode {raw:?}; expected 'time_attack' or 'gp_race'"
            ))),
        }
    }

    const fn game_mode(self) -> GameMode {
        match self {
            Self::TimeAttack => GameMode::TimeAttack,
            Self::GpRace => GameMode::GpRace,
        }
    }

    const fn menu_state(self) -> RaceStartMenuState {
        match self {
            Self::TimeAttack => RaceStartMenuState {
                selected_mode: Some(1),
                current_ghost_type: Some(0),
            },
            Self::GpRace => RaceStartMenuState {
                selected_mode: None,
                current_ghost_type: None,
            },
        }
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

#[cfg(test)]
#[path = "tests/race_start_tests.rs"]
mod tests;
