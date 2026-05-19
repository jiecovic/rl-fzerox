// rust/core/game/tests/race_start_tests.rs
use super::{
    RaceStartSetup, force_gp_race_reinit, force_time_attack_reinit, validate_gp_race_setup,
    validate_time_attack_race_setup, write_gp_race_machine_settings, write_gp_race_setup,
    write_time_attack_machine_settings, write_time_attack_menu_mode, write_time_attack_race_setup,
};
use crate::core::game::telemetry::layout::{GLOBALS, TELEMETRY_CONFIG};

#[test]
fn time_attack_setup_writes_expected_menu_and_live_fields() {
    let setup = sample_setup();
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_time_attack_machine_settings(&mut memory, setup).expect("machine settings should write");
    assert_eq!(
        read_i32(&memory, GLOBALS.selected_mode),
        1,
        "time attack should pin selected_mode"
    );
    assert_eq!(
        read_i32(&memory, GLOBALS.current_ghost_type),
        0,
        "time attack should disable ghosts in the menu globals"
    );

    write_time_attack_race_setup(&mut memory, setup).expect("race setup should write");
    validate_time_attack_race_setup(&memory, setup).expect("time attack validation should pass");
}

#[test]
fn gp_race_setup_writes_expected_menu_and_live_fields_without_ta_ghost_state() {
    let setup = sample_setup();
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_gp_race_machine_settings(&mut memory, setup).expect("machine settings should write");
    assert_eq!(
        read_i32(&memory, GLOBALS.selected_mode),
        0,
        "gp writes should leave unrelated selected_mode state untouched"
    );
    assert_eq!(
        read_i32(&memory, GLOBALS.current_ghost_type),
        0,
        "gp writes should leave unrelated ghost state untouched"
    );

    write_gp_race_setup(&mut memory, setup).expect("race setup should write");
    assert_eq!(
        read_i32(&memory, GLOBALS.difficulty),
        setup.gp_difficulty_raw_value
    );
    validate_gp_race_setup(&memory, setup).expect("gp validation should pass");
}

#[test]
fn force_reinit_sets_expected_target_game_mode() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    force_time_attack_reinit(&mut memory).expect("time attack reinit should write");
    assert_eq!(read_i32(&memory, GLOBALS.game_mode), 0x0E);
    assert_eq!(read_i32(&memory, GLOBALS.queued_game_mode), 0x0E);
    assert_eq!(read_i16(&memory, GLOBALS.game_mode_change_state), 3);

    force_gp_race_reinit(&mut memory).expect("gp reinit should write");
    assert_eq!(read_i32(&memory, GLOBALS.game_mode), 0x01);
    assert_eq!(read_i32(&memory, GLOBALS.queued_game_mode), 0x01);
    assert_eq!(read_i16(&memory, GLOBALS.game_mode_change_state), 3);
}

#[test]
fn preserve_machine_skin_leaves_existing_skin_untouched() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let mut setup = sample_setup();
    setup.machine_skin_index = -1;

    write_i16(&mut memory, GLOBALS.player_machine_skins, 111);
    write_time_attack_machine_settings(&mut memory, setup).expect("machine settings should write");
    assert_eq!(
        read_i16(&memory, GLOBALS.player_machine_skins),
        111,
        "preserve sentinel should not overwrite the selected machine skin"
    );
}

#[test]
fn time_attack_menu_mode_sets_time_attack_selection_without_full_setup() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_time_attack_menu_mode(&mut memory).expect("time attack menu mode should write");
    assert_eq!(read_i32(&memory, GLOBALS.num_players), 1);
    assert_eq!(read_i32(&memory, GLOBALS.selected_mode), 1);
    assert_eq!(read_i32(&memory, GLOBALS.current_ghost_type), 0);
}

fn sample_setup() -> RaceStartSetup {
    RaceStartSetup {
        course_index: 5,
        character_index: 7,
        machine_skin_index: 0,
        engine_setting_raw_value: 75,
        total_lap_count: 3,
        gp_difficulty_raw_value: 2,
    }
}

fn read_i32(memory: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(memory[offset..offset + 4].try_into().expect("four bytes"))
}

fn read_i16(memory: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes(memory[offset..offset + 2].try_into().expect("two bytes"))
}

fn write_i16(memory: &mut [u8], offset: usize, value: i16) {
    memory[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
