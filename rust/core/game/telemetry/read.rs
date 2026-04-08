// rust/core/game/telemetry/read.rs
//! RAM decoding helpers for building typed telemetry snapshots.

use std::mem::size_of;

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;
use crate::core::telemetry::layout::{GLOBALS, GameMode, RACER, TELEMETRY_CONFIG};
use crate::core::telemetry::model::{PlayerTelemetry, StepTelemetrySample, TelemetrySnapshot};

/// Decode a typed telemetry snapshot from a full system RAM view.
pub fn read_snapshot(system_ram: &[u8]) -> Result<TelemetrySnapshot, CoreError> {
    let player_base = validate_snapshot_memory(system_ram)?;
    let reverse_timer_offset = player_reverse_timer_offset();

    let game_mode_raw = read_u32(system_ram, GLOBALS.game_mode)?;
    let game_mode = resolve_game_mode(game_mode_raw);
    let player_state_flags = read_u32(system_ram, player_base + RACER.state_flags)?;
    let player = PlayerTelemetry {
        state_flags: player_state_flags,
        speed_kph: read_f32(system_ram, player_base + RACER.speed)? * TELEMETRY_CONFIG.speed_to_kph,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        max_energy: read_f32(system_ram, player_base + RACER.max_energy)?,
        boost_timer: read_i32(system_ram, player_base + RACER.boost_timer)?,
        reverse_timer: read_i32(system_ram, reverse_timer_offset)?,
        race_distance: read_f32(system_ram, player_base + RACER.race_distance)?,
        lap_distance: read_f32(system_ram, player_base + RACER.lap_distance)?,
        race_time_ms: read_i32(system_ram, player_base + RACER.race_time)?,
        lap: read_i16(system_ram, player_base + RACER.lap)?,
        laps_completed: read_i16(system_ram, player_base + RACER.laps_completed)?,
        position: read_i32(system_ram, player_base + RACER.position)?,
    };

    Ok(TelemetrySnapshot {
        total_lap_count: read_i32(system_ram, GLOBALS.total_lap_count)?,
        game_mode_raw,
        game_mode_name: game_mode.map_or("unknown", GameMode::wire_name),
        in_race_mode: game_mode.is_some_and(GameMode::is_race),
        total_racers: read_i32(system_ram, GLOBALS.total_racers)?,
        course_index: read_u32(system_ram, GLOBALS.course_index)?,
        player,
    })
}

pub(crate) fn read_step_sample(system_ram: &[u8]) -> Result<StepTelemetrySample, CoreError> {
    let player_base = validate_snapshot_memory(system_ram)?;
    let reverse_timer_offset = player_reverse_timer_offset();
    Ok(StepTelemetrySample {
        state_flags: read_u32(system_ram, player_base + RACER.state_flags)?,
        speed_kph: read_f32(system_ram, player_base + RACER.speed)? * TELEMETRY_CONFIG.speed_to_kph,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        race_distance: read_f32(system_ram, player_base + RACER.race_distance)?,
        reverse_timer: read_i32(system_ram, reverse_timer_offset)?,
    })
}

fn validate_snapshot_memory(system_ram: &[u8]) -> Result<usize, CoreError> {
    if system_ram.len() < TELEMETRY_CONFIG.system_ram_size_min {
        return Err(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset: 0,
            length: TELEMETRY_CONFIG.system_ram_size_min,
            available: system_ram.len(),
        });
    }

    let player_base = GLOBALS.racers + (TELEMETRY_CONFIG.player_racer_index * RACER.size);
    let player_end = player_base + RACER.position + size_of::<i32>();
    let reverse_timer_end = player_reverse_timer_offset() + size_of::<i32>();
    let required_end = player_end.max(reverse_timer_end);
    if required_end > system_ram.len() {
        return Err(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset: player_base,
            length: required_end - player_base,
            available: system_ram.len(),
        });
    }
    Ok(player_base)
}

fn player_reverse_timer_offset() -> usize {
    GLOBALS.reverse_timers + (TELEMETRY_CONFIG.player_racer_index * size_of::<i32>())
}

fn resolve_game_mode(game_mode_raw: u32) -> Option<GameMode> {
    GameMode::try_from(game_mode_raw & 0x1F).ok()
}

fn read_i16(memory: &[u8], offset: usize) -> Result<i16, CoreError> {
    Ok(i16::from_le_bytes(read_array(memory, offset)?))
}

fn read_i32(memory: &[u8], offset: usize) -> Result<i32, CoreError> {
    Ok(i32::from_le_bytes(read_array(memory, offset)?))
}

fn read_u32(memory: &[u8], offset: usize) -> Result<u32, CoreError> {
    Ok(u32::from_le_bytes(read_array(memory, offset)?))
}

fn read_f32(memory: &[u8], offset: usize) -> Result<f32, CoreError> {
    Ok(f32::from_le_bytes(read_array(memory, offset)?))
}

fn read_array<const N: usize>(memory: &[u8], offset: usize) -> Result<[u8; N], CoreError> {
    let end = offset + N;
    let bytes = memory
        .get(offset..end)
        .ok_or_else(|| memory_error(offset, N, memory.len()))?;
    let mut array = [0_u8; N];
    array.copy_from_slice(bytes);
    Ok(array)
}

fn memory_error(offset: usize, length: usize, available: usize) -> CoreError {
    CoreError::MemoryOutOfRange {
        memory_id: MEMORY_SYSTEM_RAM,
        offset,
        length,
        available,
    }
}
