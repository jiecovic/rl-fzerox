// rust/core/game/telemetry/read.rs
//! RAM decoding helpers for building typed telemetry snapshots.

use std::mem::size_of;

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;
use crate::core::game::memory::{
    read_f32, read_i8, read_i16, read_i32, read_u32, read_word_swapped_u8,
};
use crate::core::telemetry::layout::{
    CAMERA, COURSE_INFO, COURSE_SEGMENT, CameraRaceSetting, GLOBALS, GameMode, MACHINE_TABLE,
    RACER, RACER_SEGMENT_POSITION_INFO, RaceDifficulty, TELEMETRY_CONFIG,
};
use crate::core::telemetry::model::{
    MachineContextTelemetry, PlayerTelemetry, RacerGeometryTelemetry, StepTelemetrySample,
    TelemetrySnapshot,
};

#[derive(Clone, Copy, Debug)]
struct FutureLocalSegmentScanConfig {
    future_segment_count: usize,
    samples_per_segment: usize,
}

const FUTURE_LOCAL_SEGMENT_SCAN: FutureLocalSegmentScanConfig = FutureLocalSegmentScanConfig {
    future_segment_count: 3,
    samples_per_segment: 5,
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
        body_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.body_stat)?,
        boost_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.boost_stat)?,
        grip_stat: read_machine_i8(system_ram, machine_base + MACHINE_TABLE.grip_stat)?,
        weight: read_machine_i16(system_ram, machine_base + MACHINE_TABLE.weight)?,
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

fn read_racer_geometry(
    system_ram: &[u8],
    player_base: usize,
) -> Result<RacerGeometryTelemetry, CoreError> {
    let segment_info_base = player_base + RACER.segment_position_info;
    let current_segment_offset = read_current_segment_offset(system_ram, segment_info_base)?;
    let segment_index = read_segment_index(system_ram, current_segment_offset)?;
    let world_pos = read_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.pos,
    )?;
    let segment_center = read_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_pos,
    )?;
    let signed_lateral_offset =
        read_signed_lateral_offset(system_ram, player_base, segment_info_base)?;
    let current_radius_left = read_f32(system_ram, player_base + RACER.current_radius_left)?;
    let current_radius_right = read_f32(system_ram, player_base + RACER.current_radius_right)?;
    let future_local_segment = future_local_nearest_segment(
        system_ram,
        current_segment_offset,
        world_pos,
        signed_lateral_offset,
        current_radius_left,
        current_radius_right,
    )?;
    Ok(RacerGeometryTelemetry {
        segment_index,
        segment_t: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_t_value,
        )?,
        segment_length_proportion: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_length_proportion,
        )?,
        world_pos_x: world_pos.x,
        world_pos_y: world_pos.y,
        world_pos_z: world_pos.z,
        segment_center_x: segment_center.x,
        segment_center_y: segment_center.y,
        segment_center_z: segment_center.z,
        local_lateral_velocity: read_f32(system_ram, player_base + RACER.local_velocity)?,
        signed_lateral_offset,
        lateral_distance: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.distance_from_segment,
        )?,
        lateral_displacement_magnitude: read_vec3_magnitude(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement,
        )?,
        current_radius_left,
        current_radius_right,
        height_above_ground: read_f32(system_ram, player_base + RACER.height_above_ground)?,
        future_local_nearest_segment_index: future_local_segment.map(|segment| segment.index),
        future_local_nearest_segment_distance: future_local_segment
            .map_or(0.0, |segment| segment.distance),
        velocity_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.velocity)?,
        acceleration_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.acceleration)?,
        acceleration_force: read_f32(system_ram, player_base + RACER.acceleration_force)?,
        drift_attack_force: read_f32(system_ram, player_base + RACER.drift_attack_force)?,
        collision_mass: read_f32(system_ram, player_base + RACER.colliding_strength)?,
    })
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
struct NearestFutureSegment {
    index: i32,
    distance: f32,
}

fn future_local_nearest_segment(
    system_ram: &[u8],
    current_segment_offset: Option<usize>,
    world_pos: Vec3,
    signed_lateral_offset: f32,
    current_radius_left: f32,
    current_radius_right: f32,
) -> Result<Option<NearestFutureSegment>, CoreError> {
    if !outside_track_bounds(
        signed_lateral_offset,
        current_radius_left,
        current_radius_right,
    ) {
        return Ok(None);
    }

    let Some(mut segment_offset) = current_segment_offset else {
        return Ok(None);
    };
    let mut best: Option<NearestFutureSegment> = None;
    for _ in 0..=FUTURE_LOCAL_SEGMENT_SCAN.future_segment_count {
        if !valid_segment_base(system_ram, segment_offset) {
            return Ok(best);
        }
        if let Some(candidate) = nearest_sample_on_segment(system_ram, segment_offset, world_pos)? {
            best = Some(match best {
                Some(previous) if previous.distance <= candidate.distance => previous,
                _ => candidate,
            });
        }
        let Some(next_segment_offset) =
            read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.next)?
        else {
            return Ok(best);
        };
        segment_offset = next_segment_offset;
    }
    Ok(best)
}

fn outside_track_bounds(offset: f32, current_radius_left: f32, current_radius_right: f32) -> bool {
    if offset >= 0.0 {
        current_radius_left > 0.0 && offset > current_radius_left * 1.10
    } else {
        current_radius_right > 0.0 && offset < -current_radius_right * 1.10
    }
}

fn nearest_sample_on_segment(
    system_ram: &[u8],
    segment_offset: usize,
    world_pos: Vec3,
) -> Result<Option<NearestFutureSegment>, CoreError> {
    let Some(index) = read_segment_index(system_ram, Some(segment_offset))? else {
        return Ok(None);
    };

    let mut best_distance = f32::INFINITY;
    let denominator = (FUTURE_LOCAL_SEGMENT_SCAN.samples_per_segment - 1).max(1) as f32;
    for sample_index in 0..FUTURE_LOCAL_SEGMENT_SCAN.samples_per_segment {
        let t = sample_index as f32 / denominator;
        let Some(sample_pos) = sample_segment_spline_position(system_ram, segment_offset, t)?
        else {
            continue;
        };
        best_distance = best_distance.min(euclidean_distance(world_pos, sample_pos));
    }
    if best_distance.is_finite() {
        Ok(Some(NearestFutureSegment {
            index,
            distance: best_distance,
        }))
    } else {
        Ok(None)
    }
}

fn sample_segment_spline_position(
    system_ram: &[u8],
    segment_offset: usize,
    t: f32,
) -> Result<Option<Vec3>, CoreError> {
    let Some(prev_segment_offset) =
        read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.prev)?
    else {
        return Ok(None);
    };
    let Some(next_segment_offset) =
        read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.next)?
    else {
        return Ok(None);
    };
    let Some(next_next_segment_offset) =
        read_segment_pointer(system_ram, next_segment_offset + COURSE_SEGMENT.next)?
    else {
        return Ok(None);
    };
    if !valid_segment_base(system_ram, prev_segment_offset)
        || !valid_segment_base(system_ram, next_segment_offset)
        || !valid_segment_base(system_ram, next_next_segment_offset)
    {
        return Ok(None);
    }

    let prev_pos = read_segment_pos(system_ram, prev_segment_offset)?;
    let pos = read_segment_pos(system_ram, segment_offset)?;
    let next_pos = read_segment_pos(system_ram, next_segment_offset)?;
    let next_next_pos = read_segment_pos(system_ram, next_next_segment_offset)?;
    let tension = read_f32(system_ram, segment_offset + COURSE_SEGMENT.tension)?;
    let t_square = t * t;
    let t_cube = t_square * t;
    let prev_weight = (2.0 * t_square - t - t_cube) * tension;
    let current_weight = (2.0 - tension).mul_add(t_cube, (tension - 3.0) * t_square) + 1.0;
    let next_weight =
        (tension - 2.0).mul_add(t_cube, (3.0 - 2.0 * tension) * t_square) + (tension * t);
    let next_next_weight = (t_cube - t_square) * tension;
    Ok(Some(Vec3 {
        x: prev_weight.mul_add(
            prev_pos.x,
            current_weight.mul_add(
                pos.x,
                next_weight.mul_add(next_pos.x, next_next_weight * next_next_pos.x),
            ),
        ),
        y: prev_weight.mul_add(
            prev_pos.y,
            current_weight.mul_add(
                pos.y,
                next_weight.mul_add(next_pos.y, next_next_weight * next_next_pos.y),
            ),
        ),
        z: prev_weight.mul_add(
            prev_pos.z,
            current_weight.mul_add(
                pos.z,
                next_weight.mul_add(next_pos.z, next_next_weight * next_next_pos.z),
            ),
        ),
    }))
}

fn valid_segment_base(system_ram: &[u8], segment_offset: usize) -> bool {
    segment_offset + COURSE_SEGMENT.prev + size_of::<u32>() <= system_ram.len()
}

fn euclidean_distance(lhs: Vec3, rhs: Vec3) -> f32 {
    let x = lhs.x - rhs.x;
    let y = lhs.y - rhs.y;
    let z = lhs.z - rhs.z;
    x.mul_add(x, y.mul_add(y, z * z)).sqrt()
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

fn read_current_segment_offset(
    system_ram: &[u8],
    segment_info_base: usize,
) -> Result<Option<usize>, CoreError> {
    let pointer = read_u32(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.course_segment,
    )?;
    Ok(kseg0_pointer_to_offset(pointer, system_ram.len()))
}

fn read_segment_index(
    system_ram: &[u8],
    segment_offset: Option<usize>,
) -> Result<Option<i32>, CoreError> {
    let Some(segment_offset) = segment_offset else {
        return Ok(None);
    };
    let segment_index_offset = segment_offset + COURSE_SEGMENT.segment_index;
    if segment_index_offset + size_of::<i32>() > system_ram.len() {
        return Ok(None);
    }
    Ok(Some(read_i32(system_ram, segment_index_offset)?))
}

fn read_segment_pointer(system_ram: &[u8], offset: usize) -> Result<Option<usize>, CoreError> {
    if offset + size_of::<u32>() > system_ram.len() {
        return Ok(None);
    }
    Ok(kseg0_pointer_to_offset(
        read_u32(system_ram, offset)?,
        system_ram.len(),
    ))
}

fn read_segment_pos(system_ram: &[u8], segment_offset: usize) -> Result<Vec3, CoreError> {
    read_vec3(system_ram, segment_offset + COURSE_SEGMENT.pos)
}

#[derive(Clone, Copy, Debug, Default)]
struct CurrentCourseInfo {
    segment_count: i32,
    length: f32,
}

fn read_current_course_info(system_ram: &[u8]) -> Result<CurrentCourseInfo, CoreError> {
    let pointer = read_u32(system_ram, GLOBALS.current_course_info)?;
    let Some(course_info_offset) = kseg0_pointer_to_offset(pointer, system_ram.len()) else {
        return Ok(CurrentCourseInfo::default());
    };
    let segment_count_offset = course_info_offset + COURSE_INFO.segment_count;
    let length_offset = course_info_offset + COURSE_INFO.length;
    if segment_count_offset + size_of::<i32>() > system_ram.len()
        || length_offset + size_of::<f32>() > system_ram.len()
    {
        return Ok(CurrentCourseInfo::default());
    }
    Ok(CurrentCourseInfo {
        segment_count: read_i32(system_ram, segment_count_offset)?,
        length: read_f32(system_ram, length_offset)?,
    })
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

fn read_vec3(memory: &[u8], offset: usize) -> Result<Vec3, CoreError> {
    Ok(Vec3 {
        x: read_f32(memory, offset)?,
        y: read_f32(memory, offset + size_of::<f32>())?,
        z: read_f32(memory, offset + (2 * size_of::<f32>()))?,
    })
}

fn dot_vec3(memory: &[u8], lhs_offset: usize, rhs_offset: usize) -> Result<f32, CoreError> {
    let x = read_f32(memory, lhs_offset)? * read_f32(memory, rhs_offset)?;
    let y = read_f32(memory, lhs_offset + size_of::<f32>())?
        * read_f32(memory, rhs_offset + size_of::<f32>())?;
    let z = read_f32(memory, lhs_offset + (2 * size_of::<f32>()))?
        * read_f32(memory, rhs_offset + (2 * size_of::<f32>()))?;
    Ok(x + y + z)
}
