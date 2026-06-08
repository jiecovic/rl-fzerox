// rust/core/host/tests/runtime_spin_tests.rs
// Covers native spin macro sequencing without requiring a loaded core.
use libretro_sys::{DEVICE_ID_JOYPAD_A, DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R2};

use super::{SpinMacroState, SpinRequest};
use crate::core::input::ControllerState;

#[test]
fn none_request_leaves_controller_unchanged() {
    let mut spin = SpinMacroState::default();
    let mut stats = spin.begin_step(SpinRequest::None);
    let controller = ControllerState::from_normalized(1 << DEVICE_ID_JOYPAD_A, 0.0, 0.0, 0.0, 0.0);

    let frame = spin.next_controller(controller, 8);
    if frame.macro_owns_lean {
        stats.active_frames += 1;
    }

    assert!(!stats.started);
    assert!(!frame.macro_owns_lean);
    assert_eq!(frame.controller_state, controller);
    assert_eq!(spin.status().cooldown_frames, 0);
}

#[test]
fn left_spin_holds_right_lean_and_double_taps_left() {
    let mut spin = SpinMacroState::default();
    let mut stats = spin.begin_step(SpinRequest::Left);
    let controller = ControllerState::from_normalized(
        (1 << DEVICE_ID_JOYPAD_A) | (1 << DEVICE_ID_JOYPAD_L2),
        0.0,
        0.0,
        0.0,
        0.0,
    );

    let first = spin.next_controller(controller, 8);
    if first.macro_owns_lean {
        stats.active_frames += 1;
        stats.lean_owned_frames += 1;
    }
    let second_tap = spin.next_controller(controller, 8);
    if second_tap.macro_owns_lean {
        stats.active_frames += 1;
        stats.lean_owned_frames += 1;
    }
    let gap = spin.next_controller(controller, 8);
    if gap.macro_owns_lean {
        stats.active_frames += 1;
        stats.lean_owned_frames += 1;
    }

    assert!(stats.started);
    assert_ne!(first.controller_state.joypad_state(DEVICE_ID_JOYPAD_A), 0);
    assert_ne!(first.controller_state.joypad_state(DEVICE_ID_JOYPAD_R2), 0);
    assert_ne!(first.controller_state.joypad_state(DEVICE_ID_JOYPAD_L2), 0);
    assert_ne!(
        second_tap
            .controller_state
            .joypad_state(DEVICE_ID_JOYPAD_L2),
        0
    );
    assert_ne!(gap.controller_state.joypad_state(DEVICE_ID_JOYPAD_R2), 0);
    assert_eq!(gap.controller_state.joypad_state(DEVICE_ID_JOYPAD_L2), 0);
    assert_eq!(stats.active_frames, 3);
    assert_eq!(stats.lean_owned_frames, 3);
}

#[test]
fn completed_spin_enters_cooldown_and_blocks_restart() {
    let mut spin = SpinMacroState::default();
    let controller = ControllerState::default();
    assert!(spin.begin_step(SpinRequest::Right).started);

    while spin.status().active {
        spin.next_controller(controller, 8);
    }

    let cooldown = spin.status().cooldown_frames;
    assert!(cooldown > 0);
    assert!(!spin.begin_step(SpinRequest::Left).started);
    assert_eq!(spin.status().cooldown_frames, cooldown);
}

#[test]
fn completed_spin_uses_configured_cooldown_frames() {
    let mut spin = SpinMacroState::default();
    let controller = ControllerState::default();
    assert!(spin.begin_step(SpinRequest::Right).started);

    while spin.status().active {
        spin.next_controller(controller, 3);
    }

    assert_eq!(spin.status().cooldown_frames, 3);
}
