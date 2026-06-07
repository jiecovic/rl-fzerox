// rust/core/host/tests/input_tests.rs
// Covers normalized controller input clamping into libretro's integer range.
use libretro_sys::{
    DEVICE_ID_ANALOG_X, DEVICE_ID_ANALOG_Y, DEVICE_ID_JOYPAD_A, DEVICE_ID_JOYPAD_B,
    DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R2, DEVICE_ID_JOYPAD_Y, DEVICE_INDEX_ANALOG_LEFT,
    DEVICE_INDEX_ANALOG_RIGHT,
};

use super::{fzerox_menu_button_mask, ControllerState, FZeroXMenuButton, RaceControlState};

#[test]
fn controller_state_clamps_normalized_axes_to_libretro_range() {
    let state = ControllerState::from_normalized(0, -2.0, 2.0, 0.5, -0.5);

    assert_eq!(
        state.analog_state(DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_X),
        -i16::MAX
    );
    assert_eq!(
        state.analog_state(DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_Y),
        i16::MAX
    );
    assert_eq!(
        state.analog_state(DEVICE_INDEX_ANALOG_RIGHT, DEVICE_ID_ANALOG_X),
        16384
    );
    assert_eq!(
        state.analog_state(DEVICE_INDEX_ANALOG_RIGHT, DEVICE_ID_ANALOG_Y),
        -16384
    );
}

#[test]
fn race_control_state_maps_semantics_to_pinned_mupen_buttons() {
    let controls = RaceControlState {
        gas: true,
        air_brake: true,
        boost: true,
        lean_left: true,
        lean_right: true,
        stick_x: 0.0,
        pitch: 0.0,
    };
    let state = controls.to_controller_state();

    assert_ne!(state.joypad_state(DEVICE_ID_JOYPAD_B), 0);
    assert_ne!(state.joypad_state(DEVICE_ID_JOYPAD_A), 0);
    assert_ne!(state.joypad_state(DEVICE_ID_JOYPAD_Y), 0);
    assert_ne!(state.joypad_state(DEVICE_ID_JOYPAD_L2), 0);
    assert_ne!(state.joypad_state(DEVICE_ID_JOYPAD_R2), 0);
}

#[test]
fn menu_button_masks_reuse_fzerox_semantics_instead_of_retropad_names() {
    let confirm = fzerox_menu_button_mask(FZeroXMenuButton::Confirm);
    let cancel = fzerox_menu_button_mask(FZeroXMenuButton::Cancel);

    assert_eq!(
        confirm,
        RaceControlState {
            gas: true,
            ..RaceControlState::default()
        }
        .to_controller_state()
        .joypad_mask()
    );
    assert_eq!(
        cancel,
        RaceControlState {
            boost: true,
            ..RaceControlState::default()
        }
        .to_controller_state()
        .joypad_mask()
    );
    assert_ne!(confirm, 1 << DEVICE_ID_JOYPAD_A);
    assert_ne!(cancel, 1 << DEVICE_ID_JOYPAD_B);
}
