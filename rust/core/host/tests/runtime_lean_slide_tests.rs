// rust/core/host/tests/runtime_lean_slide_tests.rs

use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R2};

use super::{SLIDE_ASSIST_RAM, patch_lean_timers};
use crate::core::input::ControllerState;

#[test]
fn held_left_lean_patches_both_slide_timers() {
    let mut memory = timer_memory();
    let controller_state =
        ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_L2), 0.0, 0.0, 0.0, 0.0);

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(
        read_i16(&memory, z_button_timer_offset()),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
    assert_eq!(
        read_i16(&memory, r_button_timer_offset()),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
}

#[test]
fn held_right_lean_patches_both_slide_timers() {
    let mut memory = timer_memory();
    let controller_state =
        ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_R2), 0.0, 0.0, 0.0, 0.0);

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(
        read_i16(&memory, z_button_timer_offset()),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
    assert_eq!(
        read_i16(&memory, r_button_timer_offset()),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
}

#[test]
fn idle_leans_leave_timers_unchanged() {
    let mut memory = timer_memory();
    let controller_state = ControllerState::default();

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(read_i16(&memory, z_button_timer_offset()), 0);
    assert_eq!(read_i16(&memory, r_button_timer_offset()), 0);
}

fn button_mask(button_id: u32) -> u16 {
    1_u16 << button_id
}

fn timer_memory() -> Vec<u8> {
    vec![0_u8; r_button_timer_offset() + 2]
}

fn z_button_timer_offset() -> usize {
    SLIDE_ASSIST_RAM.timer_offsets[0]
}

fn r_button_timer_offset() -> usize {
    SLIDE_ASSIST_RAM.timer_offsets[1]
}

fn read_i16(memory: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([memory[offset], memory[offset + 1]])
}
