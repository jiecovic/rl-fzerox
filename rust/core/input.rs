// rust/core/input.rs
use libretro_sys::{
    DEVICE_ID_ANALOG_X, DEVICE_ID_ANALOG_Y, DEVICE_INDEX_ANALOG_LEFT, DEVICE_INDEX_ANALOG_RIGHT,
};

const LIBRETRO_ANALOG_MAX: f32 = 0x7fff as f32;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControllerState {
    joypad_mask: u16,
    left_stick_x: i16,
    left_stick_y: i16,
    right_stick_x: i16,
    right_stick_y: i16,
}

impl ControllerState {
    pub fn from_normalized(
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> Self {
        Self {
            joypad_mask,
            left_stick_x: normalized_axis_to_i16(left_stick_x),
            left_stick_y: normalized_axis_to_i16(left_stick_y),
            right_stick_x: normalized_axis_to_i16(right_stick_x),
            right_stick_y: normalized_axis_to_i16(right_stick_y),
        }
    }

    pub fn joypad_state(&self, button_id: u32) -> i16 {
        if button_id >= u16::BITS {
            return 0;
        }
        (((self.joypad_mask >> button_id) & 1) != 0) as i16
    }

    pub fn analog_state(&self, index: u32, id: u32) -> i16 {
        match (index, id) {
            (DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_X) => self.left_stick_x,
            (DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_Y) => self.left_stick_y,
            (DEVICE_INDEX_ANALOG_RIGHT, DEVICE_ID_ANALOG_X) => self.right_stick_x,
            (DEVICE_INDEX_ANALOG_RIGHT, DEVICE_ID_ANALOG_Y) => self.right_stick_y,
            _ => 0,
        }
    }
}

fn normalized_axis_to_i16(value: f32) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * LIBRETRO_ANALOG_MAX).round() as i16
}

#[cfg(test)]
mod tests {
    use libretro_sys::{
        DEVICE_ID_ANALOG_X, DEVICE_ID_ANALOG_Y, DEVICE_INDEX_ANALOG_LEFT, DEVICE_INDEX_ANALOG_RIGHT,
    };

    use super::ControllerState;

    #[test]
    fn controller_state_clamps_normalized_axes_to_libretro_range() {
        let state = ControllerState::from_normalized(0, -2.0, 2.0, 0.5, -0.5);

        assert_eq!(
            state.analog_state(DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_X),
            -0x7fff
        );
        assert_eq!(
            state.analog_state(DEVICE_INDEX_ANALOG_LEFT, DEVICE_ID_ANALOG_Y),
            0x7fff
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
}
