// rust/core/game/tests/race_start_tests.rs
use super::{
    RaceStartMode, RaceStartSetup, force_race_reinit, validate_race_setup, vehicle_setup_info,
    write_machine_settings, write_race_setup, write_time_attack_menu_mode,
};
use crate::core::game::telemetry::layout::{GLOBALS, RACER, TELEMETRY_CONFIG};

#[test]
fn time_attack_setup_writes_expected_menu_and_live_fields() {
    let setup = sample_setup();
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_machine_settings(&mut memory, RaceStartMode::TimeAttack, setup)
        .expect("machine settings should write");
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

    write_race_setup(&mut memory, RaceStartMode::TimeAttack, setup)
        .expect("race setup should write");
    validate_race_setup(&memory, RaceStartMode::TimeAttack, setup)
        .expect("time attack validation should pass");
}

#[test]
fn gp_race_setup_writes_expected_menu_and_live_fields_without_ta_ghost_state() {
    let setup = sample_setup();
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_machine_settings(&mut memory, RaceStartMode::GpRace, setup)
        .expect("machine settings should write");
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

    write_race_setup(&mut memory, RaceStartMode::GpRace, setup).expect("race setup should write");
    assert_eq!(
        read_i32(&memory, GLOBALS.difficulty),
        setup.gp_difficulty_raw_value
    );
    validate_race_setup(&memory, RaceStartMode::GpRace, setup).expect("gp validation should pass");
}

#[test]
fn gp_race_setup_accepts_x_cup_course_slot() {
    let mut setup = sample_setup();
    setup.course_index = 48;
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_machine_settings(&mut memory, RaceStartMode::GpRace, setup)
        .expect("x cup gp machine settings should write");
    write_race_setup(&mut memory, RaceStartMode::GpRace, setup)
        .expect("x cup gp race setup should write");

    assert_eq!(read_i32(&memory, GLOBALS.course_index), 48);
    validate_race_setup(&memory, RaceStartMode::GpRace, setup)
        .expect("x cup gp validation should pass");
}

#[test]
fn time_attack_setup_rejects_x_cup_course_slot() {
    let mut setup = sample_setup();
    setup.course_index = 48;
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    write_machine_settings(&mut memory, RaceStartMode::TimeAttack, setup)
        .expect_err("x cup slot is not a time attack course");
}

#[test]
fn force_reinit_sets_expected_target_game_mode() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];

    force_race_reinit(&mut memory, RaceStartMode::TimeAttack)
        .expect("time attack reinit should write");
    assert_eq!(read_i32(&memory, GLOBALS.game_mode), 0x0E);
    assert_eq!(read_i32(&memory, GLOBALS.queued_game_mode), 0x0E);
    assert_eq!(read_i16(&memory, GLOBALS.game_mode_change_state), 3);

    force_race_reinit(&mut memory, RaceStartMode::GpRace).expect("gp reinit should write");
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
    write_machine_settings(&mut memory, RaceStartMode::TimeAttack, setup)
        .expect("machine settings should write");
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

#[test]
fn vehicle_setup_info_keeps_menu_vehicle_when_live_racer_is_uninitialized() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let racer_base = GLOBALS.racers + (TELEMETRY_CONFIG.player_racer_index * RACER.size);

    write_i16(&mut memory, GLOBALS.player_characters, 7);
    write_i8(&mut memory, racer_base + RACER.character, -1);
    write_f32(&mut memory, GLOBALS.player_engine, 0.75);
    write_f32(&mut memory, GLOBALS.character_last_engine + (7 * 4), 0.75);

    let setup = vehicle_setup_info(&memory).expect("setup probe should return partial data");

    assert_eq!(setup.player_character_index, 7);
    assert_eq!(setup.racer_character_index, -1);
    assert_eq!(setup.engine_setting, 0.75);
    assert_eq!(setup.character_engine_setting, Some(0.75));
    assert_eq!(setup.racer_engine_curve, None);
}

#[test]
fn vehicle_setup_info_does_not_fail_when_machine_indices_are_uninitialized() {
    let mut memory = vec![0_u8; TELEMETRY_CONFIG.system_ram_size_min];
    let racer_base = GLOBALS.racers + (TELEMETRY_CONFIG.player_racer_index * RACER.size);

    write_i16(&mut memory, GLOBALS.player_characters, -1);
    write_i8(&mut memory, racer_base + RACER.character, -1);
    write_f32(&mut memory, GLOBALS.player_engine, 0.5);

    let setup = vehicle_setup_info(&memory).expect("setup probe should return partial data");

    assert_eq!(setup.player_character_index, -1);
    assert_eq!(setup.racer_character_index, -1);
    assert_eq!(setup.engine_setting, 0.5);
    assert_eq!(setup.character_engine_setting, None);
    assert_eq!(setup.racer_engine_curve, None);
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

fn write_i8(memory: &mut [u8], offset: usize, value: i8) {
    memory[offset] = value.to_le_bytes()[0];
}

fn write_f32(memory: &mut [u8], offset: usize, value: f32) {
    memory[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
