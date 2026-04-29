// rust/core/host/runtime/lean_slide/tests.rs

use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R};

use super::{R_BUTTON_TIMER_OFFSET, SLIDE_ASSIST_RAM, Z_BUTTON_TIMER_OFFSET, patch_lean_timers};
use crate::core::input::ControllerState;

#[test]
fn held_left_lean_patches_both_slide_timers() {
    let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
    let controller_state =
        ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_L2), 0.0, 0.0, 0.0, 0.0);

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(
        read_i16(&memory, Z_BUTTON_TIMER_OFFSET),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
    assert_eq!(
        read_i16(&memory, R_BUTTON_TIMER_OFFSET),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
}

#[test]
fn held_right_lean_patches_both_slide_timers() {
    let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
    let controller_state =
        ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_R), 0.0, 0.0, 0.0, 0.0);

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(
        read_i16(&memory, Z_BUTTON_TIMER_OFFSET),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
    assert_eq!(
        read_i16(&memory, R_BUTTON_TIMER_OFFSET),
        SLIDE_ASSIST_RAM.attack_timer_guard_frames,
    );
}

#[test]
fn idle_leans_leave_timers_unchanged() {
    let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
    let controller_state = ControllerState::default();

    patch_lean_timers(&mut memory, controller_state).unwrap();

    assert_eq!(read_i16(&memory, Z_BUTTON_TIMER_OFFSET), 0);
    assert_eq!(read_i16(&memory, R_BUTTON_TIMER_OFFSET), 0);
}

fn button_mask(button_id: u32) -> u16 {
    1_u16 << button_id
}

fn read_i16(memory: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([memory[offset], memory[offset + 1]])
}
