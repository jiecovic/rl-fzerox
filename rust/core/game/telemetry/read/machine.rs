// rust/core/game/telemetry/read/machine.rs
//! Machine context decoding for the local player.

use crate::core::error::CoreError;
use crate::core::game::memory::{
    read_f32, read_i8, read_i16, read_word_swapped_i8, read_word_swapped_i16,
};
use crate::core::telemetry::layout::{GLOBALS, MACHINE_TABLE, RACER};
use crate::core::telemetry::model::MachineContextTelemetry;

pub(super) fn read_machine_context(
    system_ram: &[u8],
) -> Result<MachineContextTelemetry, CoreError> {
    let character_index = resolve_machine_character_index(system_ram)?;
    let engine_setting = read_f32(system_ram, GLOBALS.player_engine)?;
    if character_index < 0 || character_index as usize >= MACHINE_TABLE.machine_count {
        return Ok(MachineContextTelemetry {
            character_index,
            engine_setting,
            ..MachineContextTelemetry::default()
        });
    }

    let machine_base =
        MACHINE_TABLE.machines + ((character_index as usize) * MACHINE_TABLE.machine_size);
    Ok(MachineContextTelemetry {
        character_index,
        body_stat: read_word_swapped_i8(system_ram, machine_base + MACHINE_TABLE.body_stat)?,
        boost_stat: read_word_swapped_i8(system_ram, machine_base + MACHINE_TABLE.boost_stat)?,
        grip_stat: read_word_swapped_i8(system_ram, machine_base + MACHINE_TABLE.grip_stat)?,
        weight: read_word_swapped_i16(system_ram, machine_base + MACHINE_TABLE.weight)?,
        engine_setting,
    })
}

fn resolve_machine_character_index(system_ram: &[u8]) -> Result<i16, CoreError> {
    let racer_character_index = read_i8(system_ram, GLOBALS.racers + RACER.character)? as i16;
    if racer_character_index >= 0 && (racer_character_index as usize) < MACHINE_TABLE.machine_count
    {
        return Ok(racer_character_index);
    }
    read_i16(system_ram, GLOBALS.player_characters)
}
