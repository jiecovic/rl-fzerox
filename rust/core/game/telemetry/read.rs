// rust/core/game/telemetry/read.rs
//! RAM decoding helpers for building typed telemetry snapshots.

use std::mem::size_of;

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;
use crate::core::game::memory::{read_f32, read_i16, read_i32, read_u32, read_word_swapped_u8};
use crate::core::telemetry::layout::{
    CameraRaceSetting, GLOBALS, GameMode, RACER, RaceDifficulty, TELEMETRY_CONFIG,
};
use crate::core::telemetry::model::{PlayerTelemetry, StepTelemetrySample, TelemetrySnapshot};

mod course;
mod geometry;
mod machine;
mod scalars;

use course::read_current_course_info;
use geometry::{read_racer_geometry, read_signed_lateral_offset};
use machine::read_machine_context;
use scalars::{
    player_camera_setting_offset, player_damage_rumble_counter_offset, player_reverse_timer_offset,
    read_vec3_magnitude, resolve_camera_setting, resolve_difficulty, resolve_game_mode,
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
    let course_info = read_current_course_info(system_ram)?;
    let player = PlayerTelemetry {
        state_flags: player_state_flags,
        speed_kph: read_f32(system_ram, player_base + RACER.speed)? * TELEMETRY_CONFIG.speed_to_kph,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        max_energy: read_f32(system_ram, player_base + RACER.max_energy)?,
        ko_star_count: read_word_swapped_u8(system_ram, GLOBALS.player_ko_stars)? as i16,
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
        menu_selected_mode_raw: read_i32(system_ram, GLOBALS.selected_mode)?,
        menu_difficulty_state_raw: read_i32(system_ram, GLOBALS.difficulty_menu_state)?,
        menu_difficulty_cursor_raw: read_i32(system_ram, GLOBALS.difficulty_menu_cursor)?,
        menu_transition_state_raw: read_i16(system_ram, GLOBALS.game_mode_change_state)?,
        menu_current_ghost_type_raw: read_i32(system_ram, GLOBALS.current_ghost_type)?,
        queued_game_mode_raw: read_i32(system_ram, GLOBALS.queued_game_mode)?,
        in_race_mode: game_mode.is_some_and(GameMode::is_race),
        total_racers: read_i32(system_ram, GLOBALS.total_racers)?,
        gp_final_rank: read_i16(system_ram, GLOBALS.player_1_overall_position)?,
        course_index: read_u32(system_ram, GLOBALS.course_index)?,
        course_segment_count: course_info.segment_count,
        course_length: course_info.length,
        player,
    })
}

pub(crate) fn read_step_sample(system_ram: &[u8]) -> Result<StepTelemetrySample, CoreError> {
    let player_base = validate_snapshot_memory(system_ram)?;
    let reverse_timer_offset = player_reverse_timer_offset();
    let segment_info_base = player_base + RACER.segment_position_info;
    Ok(StepTelemetrySample {
        state_flags: read_u32(system_ram, player_base + RACER.state_flags)?,
        speed_kph: read_f32(system_ram, player_base + RACER.speed)? * TELEMETRY_CONFIG.speed_to_kph,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        race_distance: read_f32(system_ram, player_base + RACER.race_distance)?,
        signed_lateral_offset: read_signed_lateral_offset(
            system_ram,
            player_base,
            segment_info_base,
        )?,
        current_radius_left: read_f32(system_ram, player_base + RACER.current_radius_left)?,
        current_radius_right: read_f32(system_ram, player_base + RACER.current_radius_right)?,
        height_above_ground: read_f32(system_ram, player_base + RACER.height_above_ground)?,
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
    let ko_star_count_end = GLOBALS.player_ko_stars + size_of::<u8>();
    let menu_selected_mode_end = GLOBALS.selected_mode + size_of::<i32>();
    let menu_difficulty_state_end = GLOBALS.difficulty_menu_state + size_of::<i32>();
    let menu_difficulty_cursor_end = GLOBALS.difficulty_menu_cursor + size_of::<i32>();
    let menu_transition_state_end = GLOBALS.game_mode_change_state + size_of::<i16>();
    let menu_current_ghost_type_end = GLOBALS.current_ghost_type + size_of::<i32>();
    let queued_game_mode_end = GLOBALS.queued_game_mode + size_of::<i32>();
    let player_1_overall_position_end = GLOBALS.player_1_overall_position + size_of::<i16>();
    let required_end = player_end
        .max(camera_setting_end)
        .max(reverse_timer_end)
        .max(ko_star_count_end)
        .max(menu_selected_mode_end)
        .max(menu_difficulty_state_end)
        .max(menu_difficulty_cursor_end)
        .max(menu_transition_state_end)
        .max(menu_current_ghost_type_end)
        .max(queued_game_mode_end)
        .max(player_1_overall_position_end);
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
