// Covers decoding a representative player-one race telemetry snapshot.
use super::layout::{CAMERA, GLOBALS, RACER, TELEMETRY_CONFIG};
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
    memory[GLOBALS.course_index..GLOBALS.course_index + 4].copy_from_slice(&0_u32.to_le_bytes());
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
    assert!(
        (telemetry.player.speed_kph - (123.5 * TELEMETRY_CONFIG.speed_to_kph)).abs() < f32::EPSILON
    );
    assert!((telemetry.player.energy - 92.25).abs() < f32::EPSILON);
    assert!((telemetry.player.max_energy - 100.0).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.boost_timer, 77);
    assert!((telemetry.player.recoil_tilt_magnitude - 0.5).abs() < f32::EPSILON);
    assert!((telemetry.player.race_distance - 12_345.5).abs() < f32::EPSILON);
    assert!((telemetry.player.lap_distance - 0.0).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.race_time_ms, 12_345);
    assert_eq!(telemetry.player.lap, 2);
    assert_eq!(telemetry.player.laps_completed, 1);
    assert_eq!(telemetry.player.position, 3);
    assert_eq!(telemetry.player.state_flags, (1_u32 << 20) | (1_u32 << 30));
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
