// Covers normalized controller input clamping into libretro's integer range.
use libretro_sys::{
    DEVICE_ID_ANALOG_X, DEVICE_ID_ANALOG_Y, DEVICE_INDEX_ANALOG_LEFT, DEVICE_INDEX_ANALOG_RIGHT,
};

use super::ControllerState;

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
