// rust/core/host/input.rs
//! Conversion between Python-friendly normalized controls and libretro's
//! integer input representation.

use libretro_sys::{
    DEVICE_ID_ANALOG_X, DEVICE_ID_ANALOG_Y, DEVICE_ID_JOYPAD_A, DEVICE_ID_JOYPAD_B,
    DEVICE_ID_JOYPAD_DOWN, DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_LEFT,
    DEVICE_ID_JOYPAD_R2, DEVICE_ID_JOYPAD_RIGHT, DEVICE_ID_JOYPAD_START, DEVICE_ID_JOYPAD_UP,
    DEVICE_ID_JOYPAD_Y, DEVICE_INDEX_ANALOG_LEFT, DEVICE_INDEX_ANALOG_RIGHT,
};

const LIBRETRO_ANALOG_MAX: f32 = i16::MAX as f32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaceControlMaskBits {
    gas: u16,
    air_brake: u16,
    boost: u16,
    lean_left: u16,
    lean_right: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mupen64PlusNextRaceMapping {
    gas: u32,
    air_brake: u32,
    boost: u32,
    lean_left: u32,
    lean_right: u32,
}

impl Mupen64PlusNextRaceMapping {
    fn joypad_mask(self, controls: RaceControlState) -> u16 {
        let mut mask = 0;
        if controls.gas {
            mask |= 1_u16 << self.gas;
        }
        if controls.air_brake {
            mask |= 1_u16 << self.air_brake;
        }
        if controls.boost {
            mask |= 1_u16 << self.boost;
        }
        if controls.lean_left {
            mask |= 1_u16 << self.lean_left;
        }
        if controls.lean_right {
            mask |= 1_u16 << self.lean_right;
        }
        mask
    }
}

const RACE_CONTROL_MASKS: RaceControlMaskBits = RaceControlMaskBits {
    gas: 1 << 0,
    air_brake: 1 << 1,
    boost: 1 << 2,
    lean_left: 1 << 3,
    lean_right: 1 << 4,
};

// The host pins Mupen64Plus-Next's independent C-button mapping. With that
// profile, the semantic F-Zero X controls below map to these RetroPad buttons:
// gas -> N64 A, air_brake -> N64 C-Down, boost -> N64 B,
// lean_left -> N64 L, lean_right -> N64 R.
const RACE_INPUT_MAPPING: Mupen64PlusNextRaceMapping = Mupen64PlusNextRaceMapping {
    gas: DEVICE_ID_JOYPAD_B,
    air_brake: DEVICE_ID_JOYPAD_A,
    boost: DEVICE_ID_JOYPAD_Y,
    lean_left: DEVICE_ID_JOYPAD_L2,
    lean_right: DEVICE_ID_JOYPAD_R2,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FZeroXMenuButton {
    Confirm,
    Cancel,
    Start,
    Up,
    Down,
    Left,
    Right,
}

pub fn fzerox_menu_button_mask(button: FZeroXMenuButton) -> u16 {
    let button_id = match button {
        FZeroXMenuButton::Confirm => RACE_INPUT_MAPPING.gas,
        FZeroXMenuButton::Cancel => RACE_INPUT_MAPPING.boost,
        FZeroXMenuButton::Start => DEVICE_ID_JOYPAD_START,
        FZeroXMenuButton::Up => DEVICE_ID_JOYPAD_UP,
        FZeroXMenuButton::Down => DEVICE_ID_JOYPAD_DOWN,
        FZeroXMenuButton::Left => DEVICE_ID_JOYPAD_LEFT,
        FZeroXMenuButton::Right => DEVICE_ID_JOYPAD_RIGHT,
    };
    1_u16 << button_id
}

/// Packed controller state for a single libretro polling step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControllerState {
    joypad_mask: u16,
    left_stick_x: i16,
    left_stick_y: i16,
    right_stick_x: i16,
    right_stick_y: i16,
}

impl ControllerState {
    /// Build one controller state from normalized `[-1, 1]` stick inputs and a
    /// bitmask of pressed joypad buttons.
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

    pub fn joypad_mask(&self) -> u16 {
        self.joypad_mask
    }

    pub fn race_control_mask(&self) -> u16 {
        let mut mask = 0;
        if self.joypad_state(RACE_INPUT_MAPPING.gas) != 0 {
            mask |= RACE_CONTROL_MASKS.gas;
        }
        if self.joypad_state(RACE_INPUT_MAPPING.air_brake) != 0 {
            mask |= RACE_CONTROL_MASKS.air_brake;
        }
        if self.joypad_state(RACE_INPUT_MAPPING.boost) != 0 {
            mask |= RACE_CONTROL_MASKS.boost;
        }
        if self.joypad_state(RACE_INPUT_MAPPING.lean_left) != 0 {
            mask |= RACE_CONTROL_MASKS.lean_left;
        }
        if self.joypad_state(RACE_INPUT_MAPPING.lean_right) != 0 {
            mask |= RACE_CONTROL_MASKS.lean_right;
        }
        mask
    }

    pub fn with_joypad_mask(self, joypad_mask: u16) -> Self {
        Self {
            joypad_mask,
            ..self
        }
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

/// Semantic F-Zero X gameplay controls for the repeated-step path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RaceControlState {
    pub gas: bool,
    pub air_brake: bool,
    pub boost: bool,
    pub lean_left: bool,
    pub lean_right: bool,
    pub stick_x: f32,
    pub pitch: f32,
}

impl RaceControlState {
    pub fn to_controller_state(self) -> ControllerState {
        ControllerState::from_normalized(
            RACE_INPUT_MAPPING.joypad_mask(self),
            self.stick_x,
            self.pitch,
            0.0,
            0.0,
        )
    }
}

fn normalized_axis_to_i16(value: f32) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * LIBRETRO_ANALOG_MAX).round() as i16
}

#[cfg(test)]
#[path = "tests/input_tests.rs"]
mod tests;
