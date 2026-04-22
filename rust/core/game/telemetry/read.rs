// rust/core/game/telemetry/read.rs
//! RAM decoding helpers for building typed telemetry snapshots.

use std::mem::size_of;

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;
use crate::core::game::memory::{read_f32, read_i16, read_i32, read_u32, read_word_swapped_u8};
use crate::core::telemetry::layout::{
    CAMERA, COURSE_INFO, COURSE_SEGMENT, CameraRaceSetting, GLOBALS, GameMode, MACHINE_TABLE,
    RACER, RACER_SEGMENT_POSITION_INFO, RaceDifficulty, TELEMETRY_CONFIG,
};
use crate::core::telemetry::model::{
    MachineContextTelemetry, PlayerTelemetry, RacerGeometryTelemetry, StepTelemetrySample,
    TelemetrySnapshot,
};

/// Decode a typed telemetry snapshot from a full system RAM view.
pub fn read_snapshot(system_ram: &[u8]) -> Result<TelemetrySnapshot, CoreError> {
    let player_base = validate_snapshot_memory(system_ram)?;
    let reverse_timer_offset = player_reverse_timer_offset();

    let game_mode_raw = read_u32(system_ram, GLOBALS.game_mode)?;
    let game_mode = resolve_game_mode(game_mode_raw);
    let difficulty_raw = read_i32(system_ram, GLOBALS.difficulty)?;
    let difficulty = resolve_difficulty(difficulty_raw);
    let camera_setting_raw = read_i32(system_ram, player_camera_setting_offset())?;
    let camera_setting = resolve_camera_setting(camera_setting_raw);
    let player_state_flags = read_u32(system_ram, player_base + RACER.state_flags)?;
    let player = PlayerTelemetry {
        state_flags: player_state_flags,
        speed_kph: read_f32(system_ram, player_base + RACER.speed)? * TELEMETRY_CONFIG.speed_to_kph,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        max_energy: read_f32(system_ram, player_base + RACER.max_energy)?,
        boost_timer: read_i32(system_ram, player_base + RACER.boost_timer)?,
        recoil_tilt_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.recoil_tilt)?,
        damage_rumble_counter: read_i32(system_ram, player_damage_rumble_counter_offset())?,
        reverse_timer: read_i32(system_ram, reverse_timer_offset)?,
        race_distance: read_f32(system_ram, player_base + RACER.race_distance)?,
        lap_distance: read_f32(system_ram, player_base + RACER.lap_distance)?,
        race_time_ms: read_i32(system_ram, player_base + RACER.race_time)?,
        lap: read_i16(system_ram, player_base + RACER.lap)?,
        laps_completed: read_i16(system_ram, player_base + RACER.laps_completed)?,
        position: read_i32(system_ram, player_base + RACER.position)?,
        geometry: read_racer_geometry(system_ram, player_base)?,
        machine_context: read_machine_context(system_ram)?,
    };

    Ok(TelemetrySnapshot {
        total_lap_count: read_i32(system_ram, GLOBALS.total_lap_count)?,
        difficulty_raw,
        difficulty_name: difficulty.map_or("unknown", RaceDifficulty::wire_name),
        camera_setting_raw,
        camera_setting_name: camera_setting.map_or("unknown", CameraRaceSetting::wire_name),
        race_intro_timer: read_i32(system_ram, GLOBALS.race_intro_timer)?,
        game_mode_raw,
        game_mode_name: game_mode.map_or("unknown", GameMode::wire_name),
        in_race_mode: game_mode.is_some_and(GameMode::is_race),
        total_racers: read_i32(system_ram, GLOBALS.total_racers)?,
        course_index: read_u32(system_ram, GLOBALS.course_index)?,
        course_length: read_current_course_length(system_ram)?,
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
        damage_rumble_counter: read_i32(system_ram, player_damage_rumble_counter_offset())?,
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
    let camera_setting_end = player_camera_setting_offset() + size_of::<i32>();
    let reverse_timer_end = player_reverse_timer_offset() + size_of::<i32>();
    let required_end = player_end.max(camera_setting_end).max(reverse_timer_end);
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

fn player_damage_rumble_counter_offset() -> usize {
    GLOBALS.damage_rumble_counters + (TELEMETRY_CONFIG.player_racer_index * size_of::<i32>())
}

fn player_camera_setting_offset() -> usize {
    GLOBALS.cameras + CAMERA.race_setting
}

fn read_machine_context(system_ram: &[u8]) -> Result<MachineContextTelemetry, CoreError> {
    let character_index = read_i16(system_ram, GLOBALS.player_characters)?;
    let engine_setting = read_f32(system_ram, GLOBALS.player_engine)?;
    if character_index < 0 || character_index as usize >= MACHINE_TABLE.machine_count {
        return Ok(MachineContextTelemetry {
            engine_setting,
            ..MachineContextTelemetry::default()
        });
    }

    let machine_base =
        MACHINE_TABLE.machines + ((character_index as usize) * MACHINE_TABLE.machine_size);
    Ok(MachineContextTelemetry {
        body_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.body_stat)?,
        boost_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.boost_stat)?,
        grip_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.grip_stat)?,
        weight: read_machine_i16(system_ram, machine_base + MACHINE_TABLE.weight)?,
        engine_setting,
    })
}

fn read_racer_geometry(
    system_ram: &[u8],
    player_base: usize,
) -> Result<RacerGeometryTelemetry, CoreError> {
    let segment_info_base = player_base + RACER.segment_position_info;
    Ok(RacerGeometryTelemetry {
        segment_index: read_current_segment_index(system_ram, segment_info_base)?,
        segment_t: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_t_value,
        )?,
        segment_length_proportion: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_length_proportion,
        )?,
        local_lateral_velocity: read_f32(system_ram, player_base + RACER.local_velocity)?,
        signed_lateral_offset: read_signed_lateral_offset(
            system_ram,
            player_base,
            segment_info_base,
        )?,
        lateral_distance: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.distance_from_segment,
        )?,
        lateral_displacement_magnitude: read_vec3_magnitude(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement,
        )?,
        current_radius_left: read_f32(system_ram, player_base + RACER.current_radius_left)?,
        current_radius_right: read_f32(system_ram, player_base + RACER.current_radius_right)?,
        height_above_ground: read_f32(system_ram, player_base + RACER.height_above_ground)?,
        velocity_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.velocity)?,
        acceleration_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.acceleration)?,
        acceleration_force: read_f32(system_ram, player_base + RACER.acceleration_force)?,
        drift_attack_force: read_f32(system_ram, player_base + RACER.drift_attack_force)?,
        collision_mass: read_f32(system_ram, player_base + RACER.colliding_strength)?,
    })
}

fn read_signed_lateral_offset(
    system_ram: &[u8],
    player_base: usize,
    segment_info_base: usize,
) -> Result<f32, CoreError> {
    // The decomp's edge checks project segment displacement onto segmentBasis.z.
    // Positive is left of the spline centerline; negative is right.
    dot_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement,
        player_base + RACER.segment_basis + 0x18,
    )
}

fn read_current_segment_index(
    system_ram: &[u8],
    segment_info_base: usize,
) -> Result<Option<i32>, CoreError> {
    let pointer = read_u32(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.course_segment,
    )?;
    let Some(segment_offset) = kseg0_pointer_to_offset(pointer, system_ram.len()) else {
        return Ok(None);
    };
    let segment_index_offset = segment_offset + COURSE_SEGMENT.segment_index;
    if segment_index_offset + size_of::<i32>() > system_ram.len() {
        return Ok(None);
    }
    Ok(Some(read_i32(system_ram, segment_index_offset)?))
}

fn read_current_course_length(system_ram: &[u8]) -> Result<f32, CoreError> {
    let pointer = read_u32(system_ram, GLOBALS.current_course_info)?;
    let Some(course_info_offset) = kseg0_pointer_to_offset(pointer, system_ram.len()) else {
        return Ok(0.0);
    };
    let length_offset = course_info_offset + COURSE_INFO.length;
    if length_offset + size_of::<f32>() > system_ram.len() {
        return Ok(0.0);
    }
    read_f32(system_ram, length_offset)
}

fn kseg0_pointer_to_offset(pointer: u32, memory_len: usize) -> Option<usize> {
    let address = pointer as usize;
    if address < TELEMETRY_CONFIG.kseg0_base {
        return None;
    }
    let offset = address - TELEMETRY_CONFIG.kseg0_base;
    (offset < memory_len).then_some(offset)
}

fn resolve_game_mode(game_mode_raw: u32) -> Option<GameMode> {
    GameMode::try_from(game_mode_raw & 0x1F).ok()
}

fn resolve_difficulty(difficulty_raw: i32) -> Option<RaceDifficulty> {
    RaceDifficulty::try_from(difficulty_raw).ok()
}

fn resolve_camera_setting(camera_setting_raw: i32) -> Option<CameraRaceSetting> {
    CameraRaceSetting::try_from(camera_setting_raw).ok()
}

fn read_machine_i8(memory: &[u8], offset: usize) -> Result<i8, CoreError> {
    Ok(read_word_swapped_u8(memory, offset)? as i8)
}

fn read_machine_i16(memory: &[u8], offset: usize) -> Result<i16, CoreError> {
    Ok(i16::from_be_bytes([
        read_word_swapped_u8(memory, offset)?,
        read_word_swapped_u8(memory, offset + 1)?,
    ]))
}

fn read_vec3_magnitude(memory: &[u8], offset: usize) -> Result<f32, CoreError> {
    let x = read_f32(memory, offset)?;
    let y = read_f32(memory, offset + size_of::<f32>())?;
    let z = read_f32(memory, offset + (2 * size_of::<f32>()))?;
    Ok((x.mul_add(x, y.mul_add(y, z * z))).sqrt())
}

fn dot_vec3(memory: &[u8], lhs_offset: usize, rhs_offset: usize) -> Result<f32, CoreError> {
    let x = read_f32(memory, lhs_offset)? * read_f32(memory, rhs_offset)?;
    let y = read_f32(memory, lhs_offset + size_of::<f32>())?
        * read_f32(memory, rhs_offset + size_of::<f32>())?;
    let z = read_f32(memory, lhs_offset + (2 * size_of::<f32>()))?
        * read_f32(memory, rhs_offset + (2 * size_of::<f32>()))?;
    Ok(x + y + z)
}
