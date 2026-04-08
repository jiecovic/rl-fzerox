// Covers decoding a representative player-one race telemetry snapshot.
use super::layout::{GLOBALS, RACER, RacerStateFlag, TELEMETRY_CONFIG};
use super::read_snapshot;

#[test]
fn read_snapshot_decodes_player_one_race_values() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let player_base = GLOBALS.racers;
    memory[GLOBALS.game_frame_count..GLOBALS.game_frame_count + 4]
        .copy_from_slice(&321_u32.to_le_bytes());
    memory[GLOBALS.game_mode..GLOBALS.game_mode + 4].copy_from_slice(&1_u32.to_le_bytes());
    memory[GLOBALS.course_index..GLOBALS.course_index + 4].copy_from_slice(&0_u32.to_le_bytes());
    memory[player_base + RACER.state_flags..player_base + RACER.state_flags + 4].copy_from_slice(
        &((RacerStateFlag::CanBoost as u32) | (RacerStateFlag::Active as u32)).to_le_bytes(),
    );
    memory[player_base + RACER.speed..player_base + RACER.speed + 4]
        .copy_from_slice(&123.5_f32.to_le_bytes());
    memory[player_base + RACER.energy..player_base + RACER.energy + 4]
        .copy_from_slice(&92.25_f32.to_le_bytes());
    memory[player_base + RACER.max_energy..player_base + RACER.max_energy + 4]
        .copy_from_slice(&100.0_f32.to_le_bytes());
    memory[player_base + RACER.race_distance..player_base + RACER.race_distance + 4]
        .copy_from_slice(&12_345.5_f32.to_le_bytes());
    memory[player_base + RACER.race_distance_position
        ..player_base + RACER.race_distance_position + 4]
        .copy_from_slice(&12_340.0_f32.to_le_bytes());
    memory[player_base + RACER.race_time..player_base + RACER.race_time + 4]
        .copy_from_slice(&12_345_i32.to_le_bytes());
    memory[player_base + RACER.lap..player_base + RACER.lap + 2]
        .copy_from_slice(&2_i16.to_le_bytes());
    memory[player_base + RACER.position..player_base + RACER.position + 4]
        .copy_from_slice(&3_i32.to_le_bytes());
    memory[player_base + RACER.character] = 0;
    memory[player_base + RACER.machine_index] = 7;

    let telemetry = read_snapshot(&memory).expect("telemetry should decode");

    assert_eq!(telemetry.system_ram_size, memory.len());
    assert_eq!(telemetry.game_frame_count, 321);
    assert_eq!(telemetry.game_mode_raw, 1);
    assert_eq!(telemetry.game_mode_name, "gp_race");
    assert!(telemetry.in_race_mode);
    assert_eq!(telemetry.course_index, 0);
    assert!((telemetry.player.speed_raw - 123.5).abs() < f32::EPSILON);
    assert!(
        (telemetry.player.speed_kph - (123.5 * TELEMETRY_CONFIG.speed_to_kph)).abs() < f32::EPSILON
    );
    assert!((telemetry.player.energy - 92.25).abs() < f32::EPSILON);
    assert!((telemetry.player.max_energy - 100.0).abs() < f32::EPSILON);
    assert!((telemetry.player.race_distance - 12_345.5).abs() < f32::EPSILON);
    assert!((telemetry.player.race_distance_position - 12_340.0).abs() < f32::EPSILON);
    assert_eq!(telemetry.player.race_time_ms, 12_345);
    assert_eq!(telemetry.player.lap, 2);
    assert_eq!(telemetry.player.position, 3);
    assert_eq!(telemetry.player.character, 0);
    assert_eq!(telemetry.player.machine_index, 7);
    assert_eq!(telemetry.player.state_labels, vec!["can_boost", "active"]);
}
