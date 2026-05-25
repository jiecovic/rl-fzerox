// rust/core/host/runtime/spin.rs
//! Native spin-attack macro state.
//!
//! F-Zero X accepts a spin attack as "hold one lean button, double-tap the
//! other". The policy exposes that as one high-level request while this module
//! owns the frame-level button sequence inside the native repeated-step loop.

use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R};

use crate::core::input::ControllerState;

use super::step::StepSpinStatus;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpinRequest {
    #[default]
    None,
    Left,
    Right,
}

impl SpinRequest {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "left" => Some(Self::Left),
            "right" => Some(Self::Right),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SpinStepStats {
    pub(super) started: bool,
    pub(super) active_frames: usize,
    pub(super) lean_owned_frames: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LeanButtonMasks {
    left: u16,
    right: u16,
}

impl LeanButtonMasks {
    const fn all(self) -> u16 {
        self.left | self.right
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpinMacroTiming {
    tap_frames: usize,
    gap_frames: usize,
    settle_frames: usize,
    cooldown_frames: usize,
}

impl SpinMacroTiming {
    const fn total_frames(self) -> usize {
        self.tap_frames * 2 + self.gap_frames + self.settle_frames
    }

    const fn second_tap_start(self) -> usize {
        self.tap_frames + self.gap_frames
    }

    fn tap_button_pressed(self, frame_index: usize) -> bool {
        frame_index < self.tap_frames
            || (frame_index >= self.second_tap_start()
                && frame_index < self.second_tap_start() + self.tap_frames)
    }
}

const LEAN_BUTTONS: LeanButtonMasks = LeanButtonMasks {
    left: 1_u16 << DEVICE_ID_JOYPAD_L2,
    right: 1_u16 << DEVICE_ID_JOYPAD_R,
};

const SPIN_TIMING: SpinMacroTiming = SpinMacroTiming {
    tap_frames: 2,
    gap_frames: 2,
    settle_frames: 2,
    cooldown_frames: 8,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpinDirection {
    Left,
    Right,
}

impl SpinDirection {
    fn from_request(request: SpinRequest) -> Option<Self> {
        match request {
            SpinRequest::None => None,
            SpinRequest::Left => Some(Self::Left),
            SpinRequest::Right => Some(Self::Right),
        }
    }

    fn button_masks(self) -> SpinButtonPair {
        match self {
            // A left spin uses the left-side attack tap while the opposite
            // lean button is held. Right spin mirrors the sequence.
            Self::Left => SpinButtonPair {
                held: LEAN_BUTTONS.right,
                tapped: LEAN_BUTTONS.left,
            },
            Self::Right => SpinButtonPair {
                held: LEAN_BUTTONS.left,
                tapped: LEAN_BUTTONS.right,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpinButtonPair {
    held: u16,
    tapped: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveSpinMacro {
    direction: SpinDirection,
    frame_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SpinFrameController {
    pub(super) controller_state: ControllerState,
    pub(super) macro_owns_lean: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SpinMacroState {
    active: Option<ActiveSpinMacro>,
    cooldown_frames: usize,
}

impl SpinMacroState {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(super) fn begin_step(&mut self, request: SpinRequest) -> SpinStepStats {
        let mut stats = SpinStepStats::default();
        if self.active.is_none()
            && self.cooldown_frames == 0
            && let Some(direction) = SpinDirection::from_request(request)
        {
            self.active = Some(ActiveSpinMacro {
                direction,
                frame_index: 0,
            });
            stats.started = true;
        }
        stats
    }

    pub(super) fn next_controller(&mut self, base: ControllerState) -> SpinFrameController {
        let Some(active) = self.active else {
            self.cooldown_frames = self.cooldown_frames.saturating_sub(1);
            return SpinFrameController {
                controller_state: base,
                macro_owns_lean: false,
            };
        };

        let button_pair = active.direction.button_masks();
        let tap_mask = if SPIN_TIMING.tap_button_pressed(active.frame_index) {
            button_pair.tapped
        } else {
            0
        };
        let next_mask = (base.joypad_mask() & !LEAN_BUTTONS.all()) | button_pair.held | tap_mask;
        let next_frame_index = active.frame_index + 1;
        if next_frame_index >= SPIN_TIMING.total_frames() {
            self.active = None;
            self.cooldown_frames = SPIN_TIMING.cooldown_frames;
        } else {
            self.active = Some(ActiveSpinMacro {
                direction: active.direction,
                frame_index: next_frame_index,
            });
        }
        SpinFrameController {
            controller_state: base.with_joypad_mask(next_mask),
            macro_owns_lean: true,
        }
    }

    pub(super) fn status(&self) -> StepSpinStatus {
        StepSpinStatus {
            active: self.active.is_some(),
            frames_remaining: self
                .active
                .map(|active| SPIN_TIMING.total_frames() - active.frame_index)
                .unwrap_or(0),
            cooldown_frames: self.cooldown_frames,
        }
    }
}

#[cfg(test)]
#[path = "../tests/runtime_spin_tests.rs"]
mod tests;
