// Covers decoding a representative player-one race telemetry snapshot.
use super::layout::{
    CAMERA, COURSE_INFO, COURSE_SEGMENT, GLOBALS, MACHINE_TABLE, RACER,
    RACER_SEGMENT_POSITION_INFO, TELEMETRY_CONFIG,
};
use super::{read_snapshot, read_step_sample};

#[test]
fn read_snapshot_decodes_player_one_race_values() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let player_base = GLOBALS.racers;
    memory[GLOBALS.total_lap_count..GLOBALS.total_lap_count + 4]
        .copy_from_slice(&3_i32.to_le_bytes());
    memory[GLOBALS.difficulty..GLOBALS.difficulty + 4].copy_from_slice(&2_i32.to_le_bytes());
    memory[GLOBALS.race_intro_timer..GLOBALS.race_intro_timer + 4]
        .copy_from_slice(&39_i32.to_le_bytes());
    memory[GLOBALS.game_mode..GLOBALS.game_mode + 4].copy_from_slice(&1_u32.to_le_bytes());
    memory[GLOBALS.total_racers..GLOBALS.total_racers + 4].copy_from_slice(&30_i32.to_le_bytes());
    memory[GLOBALS.player_characters..GLOBALS.player_characters + 2]
        .copy_from_slice(&4_i16.to_le_bytes());
    memory[GLOBALS.player_engine..GLOBALS.player_engine + 4]
        .copy_from_slice(&0.7_f32.to_le_bytes());
    let machine_base = MACHINE_TABLE.machines + (4 * MACHINE_TABLE.machine_size);
    write_machine_u8(&mut memory, machine_base + MACHINE_TABLE.body_stat, 4);
    write_machine_u8(&mut memory, machine_base + MACHINE_TABLE.boost_stat, 3);
    write_machine_u8(&mut memory, machine_base + MACHINE_TABLE.grip_stat, 2);
    write_machine_i16(&mut memory, machine_base + MACHINE_TABLE.weight, 1260);
    memory[GLOBALS.course_index..GLOBALS.course_index + 4].copy_from_slice(&0_u32.to_le_bytes());
    let course_info_address = 0x802A_6B40_u32;
    let course_info_offset = (course_info_address as usize) - TELEMETRY_CONFIG.kseg0_base;
    memory[GLOBALS.current_course_info..GLOBALS.current_course_info + 4]
        .copy_from_slice(&course_info_address.to_le_bytes());
    memory[course_info_offset + COURSE_INFO.length..course_info_offset + COURSE_INFO.length + 4]
        .copy_from_slice(&80_000.0_f32.to_le_bytes());
    memory[GLOBALS.cameras + CAMERA.race_setting..GLOBALS.cameras + CAMERA.race_setting + 4]
        .copy_from_slice(&3_i32.to_le_bytes());
    memory[player_base + RACER.state_flags..player_base + RACER.state_flags + 4]
        .copy_from_slice(&((1_u32 << 20) | (1_u32 << 30)).to_le_bytes());
    memory[player_base + RACER.speed..player_base + RACER.speed + 4]
        .copy_from_slice(&123.5_f32.to_le_bytes());
    memory[player_base + RACER.energy..player_base + RACER.energy + 4]
        .copy_from_slice(&92.25_f32.to_le_bytes());
    memory[player_base + RACER.max_energy..player_base + RACER.max_energy + 4]
        .copy_from_slice(&100.0_f32.to_le_bytes());
    // `boostTimer` is immediately before spin-out/body-white timers; keep a
    // distinct neighbor sentinel so manual boost does not get wired to spin-out.
    memory[player_base + 0x218..player_base + 0x218 + 4].copy_from_slice(&77_i32.to_le_bytes());
    memory[player_base + 0x21C..player_base + 0x21C + 4].copy_from_slice(&999_i32.to_le_bytes());
    memory[player_base + RACER.recoil_tilt..player_base + RACER.recoil_tilt + 4]
        .copy_from_slice(&0.3_f32.to_le_bytes());
    memory[player_base + RACER.recoil_tilt + 4..player_base + RACER.recoil_tilt + 8]
        .copy_from_slice(&0.4_f32.to_le_bytes());
    memory[GLOBALS.damage_rumble_counters..GLOBALS.damage_rumble_counters + 4]
        .copy_from_slice(&1_i32.to_le_bytes());
    memory[player_base + RACER.race_distance..player_base + RACER.race_distance + 4]
        .copy_from_slice(&12_345.5_f32.to_le_bytes());
    memory[player_base + RACER.race_time..player_base + RACER.race_time + 4]
        .copy_from_slice(&12_345_i32.to_le_bytes());
    memory[player_base + RACER.lap..player_base + RACER.lap + 2]
        .copy_from_slice(&2_i16.to_le_bytes());
    memory[player_base + RACER.laps_completed..player_base + RACER.laps_completed + 2]
        .copy_from_slice(&1_i16.to_le_bytes());
    memory[player_base + RACER.position..player_base + RACER.position + 4]
        .copy_from_slice(&3_i32.to_le_bytes());
    let segment_address = 0x802C_2020_u32;
    let segment_offset = (segment_address as usize) - TELEMETRY_CONFIG.kseg0_base;
    let segment_info_base = player_base + RACER.segment_position_info;
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.course_segment
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.course_segment + 4]
        .copy_from_slice(&segment_address.to_le_bytes());
    memory[segment_offset + COURSE_SEGMENT.segment_index
        ..segment_offset + COURSE_SEGMENT.segment_index + 4]
        .copy_from_slice(&12_i32.to_le_bytes());
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_t_value
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_t_value + 4]
        .copy_from_slice(&0.25_f32.to_le_bytes());
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_length_proportion
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_length_proportion + 4]
        .copy_from_slice(&0.75_f32.to_le_bytes());
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement + 4]
        .copy_from_slice(&3.0_f32.to_le_bytes());
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement + 4
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement + 8]
        .copy_from_slice(&4.0_f32.to_le_bytes());
    memory[player_base + RACER.local_velocity..player_base + RACER.local_velocity + 4]
        .copy_from_slice(&(-9.5_f32).to_le_bytes());
    memory[player_base + RACER.segment_basis + 0x18..player_base + RACER.segment_basis + 0x1C]
        .copy_from_slice(&1.0_f32.to_le_bytes());
    memory[segment_info_base + RACER_SEGMENT_POSITION_INFO.distance_from_segment
        ..segment_info_base + RACER_SEGMENT_POSITION_INFO.distance_from_segment + 4]
        .copy_from_slice(&5.0_f32.to_le_bytes());
    memory[player_base + RACER.current_radius_left..player_base + RACER.current_radius_left + 4]
        .copy_from_slice(&100.0_f32.to_le_bytes());
    memory[player_base + RACER.current_radius_right..player_base + RACER.current_radius_right + 4]
        .copy_from_slice(&120.0_f32.to_le_bytes());
    memory[player_base + RACER.height_above_ground..player_base + RACER.height_above_ground + 4]
        .copy_from_slice(&7.0_f32.to_le_bytes());
    memory[player_base + RACER.velocity..player_base + RACER.velocity + 4]
        .copy_from_slice(&6.0_f32.to_le_bytes());
    memory[player_base + RACER.velocity + 8..player_base + RACER.velocity + 12]
        .copy_from_slice(&8.0_f32.to_le_bytes());
    memory[player_base + RACER.acceleration..player_base + RACER.acceleration + 4]
        .copy_from_slice(&1.0_f32.to_le_bytes());
    memory[player_base + RACER.acceleration + 4..player_base + RACER.acceleration + 8]
        .copy_from_slice(&2.0_f32.to_le_bytes());
    memory[player_base + RACER.acceleration + 8..player_base + RACER.acceleration + 12]
        .copy_from_slice(&2.0_f32.to_le_bytes());
    memory[player_base + RACER.acceleration_force..player_base + RACER.acceleration_force + 4]
        .copy_from_slice(&9.0_f32.to_le_bytes());
    memory[player_base + RACER.drift_attack_force..player_base + RACER.drift_attack_force + 4]
        .copy_from_slice(&10.0_f32.to_le_bytes());
    memory[player_base + RACER.colliding_strength..player_base + RACER.colliding_strength + 4]
        .copy_from_slice(&0.5_f32.to_le_bytes());

    let telemetry = read_snapshot(&memory).expect("telemetry should decode");

    assert_eq!(telemetry.game_mode_raw, 1);
    assert_eq!(telemetry.total_lap_count, 3);
    assert_eq!(telemetry.difficulty_raw, 2);
    assert_eq!(telemetry.difficulty_name, "expert");
    assert_eq!(telemetry.race_intro_timer, 39);
    assert_eq!(telemetry.camera_setting_raw, 3);
    assert_eq!(telemetry.camera_setting_name, "wide");
    assert_eq!(telemetry.total_racers, 30);
    assert_eq!(telemetry.course_index, 0);
    assert!((telemetry.course_length - 80_000.0).abs() < f32::EPSILON);
    assert!(
        (telemetry.player.speed_kph - (123.5 * TELEMETRY_CONFIG.speed_to_kph)).abs() < f32::EPSILON
    );
    assert!((telemetry.player.energy - 92.25).abs() < f32::EPSILON);
    assert!((telemetry.player.max_energy - 100.0).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.boost_timer, 77);
    assert!((telemetry.player.recoil_tilt_magnitude - 0.5).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.damage_rumble_counter, 1);
    assert!((telemetry.player.race_distance - 12_345.5).abs() < f32::EPSILON);
    assert!((telemetry.player.lap_distance - 0.0).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.race_time_ms, 12_345);
    assert_eq!(telemetry.player.lap, 2);
    assert_eq!(telemetry.player.laps_completed, 1);
    assert_eq!(telemetry.player.position, 3);
    assert_eq!(telemetry.player.state_flags, (1_u32 << 20) | (1_u32 << 30));
    assert_eq!(telemetry.player.geometry.segment_index, Some(12));
    assert!((telemetry.player.geometry.segment_t - 0.25).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.segment_length_proportion - 0.75).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.local_lateral_velocity + 9.5).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.signed_lateral_offset - 3.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.lateral_distance - 5.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.lateral_displacement_magnitude - 5.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.current_radius_left - 100.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.current_radius_right - 120.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.height_above_ground - 7.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.velocity_magnitude - 10.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.acceleration_magnitude - 3.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.acceleration_force - 9.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.drift_attack_force - 10.0).abs() < f32::EPSILON);
    assert!((telemetry.player.geometry.collision_mass - 0.5).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.machine_context.body_stat, 4);
    assert_eq!(telemetry.player.machine_context.boost_stat, 3);
    assert_eq!(telemetry.player.machine_context.grip_stat, 2);
    assert_eq!(telemetry.player.machine_context.weight, 1260);
    assert!((telemetry.player.machine_context.engine_setting - 0.7).abs() < f32::EPSILON);
}

#[test]
fn read_step_sample_decodes_player_damage_rumble_counter() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let player_base = GLOBALS.racers;
    memory[player_base + RACER.speed..player_base + RACER.speed + 4]
        .copy_from_slice(&123.5_f32.to_le_bytes());
    memory[player_base + RACER.energy..player_base + RACER.energy + 4]
        .copy_from_slice(&92.25_f32.to_le_bytes());
    memory[player_base + RACER.race_distance..player_base + RACER.race_distance + 4]
        .copy_from_slice(&12_345.5_f32.to_le_bytes());
    memory[GLOBALS.damage_rumble_counters..GLOBALS.damage_rumble_counters + 4]
        .copy_from_slice(&1_i32.to_le_bytes());

    let telemetry = read_step_sample(&memory).expect("step telemetry should decode");

    assert_eq!(telemetry.damage_rumble_counter, 1);
}

fn write_machine_u8(memory: &mut [u8], logical_offset: usize, value: u8) {
    memory[logical_offset ^ 0x03] = value;
}

fn write_machine_i16(memory: &mut [u8], logical_offset: usize, value: i16) {
    let bytes = value.to_be_bytes();
    write_machine_u8(memory, logical_offset, bytes[0]);
    write_machine_u8(memory, logical_offset + 1, bytes[1]);
}
