// rust/core/host/runtime/shoulder_slide.rs
//! RAM-assisted shoulder slide handling for F-Zero X.
//!
//! F-Zero X uses a short Z/R timer window to detect double-tap side attacks.
//! When the policy requests a slide, keep normal controller input semantics but
//! patch the slide timers past that window before the frame update. That lets
//! the policy release early without the next shoulder edge turning into a dash.

use std::mem::size_of;

use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R, MEMORY_SYSTEM_RAM};

use super::host::Host;
use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::telemetry::{player_r_button_timer_offset, player_z_button_timer_offset};

const Z_BUTTON_TIMER_OFFSET: usize = player_z_button_timer_offset();
const R_BUTTON_TIMER_OFFSET: usize = player_r_button_timer_offset();
const SLIDE_ATTACK_TIMER_GUARD_FRAMES: i16 = 15;

const SLIDE_SHOULDER_BUTTON_IDS: [u32; 2] = [DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R];
const SLIDE_TIMER_OFFSETS: [usize; 2] = [Z_BUTTON_TIMER_OFFSET, R_BUTTON_TIMER_OFFSET];

impl Host {
    pub(super) fn patch_shoulder_timers_for_slide_assist(
        &mut self,
        controller_state: ControllerState,
    ) -> Result<(), CoreError> {
        if !has_held_slide_shoulder(controller_state) {
            return Ok(());
        }
        let system_ram = self.system_ram_slice_mut()?;
        patch_shoulder_timers(system_ram, controller_state)
    }
}

fn has_held_slide_shoulder(controller_state: ControllerState) -> bool {
    SLIDE_SHOULDER_BUTTON_IDS
        .iter()
        .any(|button_id| controller_state.joypad_state(*button_id) != 0)
}

fn patch_shoulder_timers(
    system_ram: &mut [u8],
    controller_state: ControllerState,
) -> Result<(), CoreError> {
    if !has_held_slide_shoulder(controller_state) {
        return Ok(());
    }

    // The two timers are adjacent `s16` fields. Guard both together so the
    // slide primitive remains safe even on byte/halfword-swapped RDRAM views.
    for offset in SLIDE_TIMER_OFFSETS {
        write_i16(system_ram, offset, SLIDE_ATTACK_TIMER_GUARD_FRAMES)?;
    }
    Ok(())
}

fn write_i16(system_ram: &mut [u8], offset: usize, value: i16) -> Result<(), CoreError> {
    let length = size_of::<i16>();
    let end = offset
        .checked_add(length)
        .ok_or(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset,
            length,
            available: system_ram.len(),
        })?;
    if end > system_ram.len() {
        return Err(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset,
            length,
            available: system_ram.len(),
        });
    }
    system_ram[offset..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R};

    use super::{
        R_BUTTON_TIMER_OFFSET, SLIDE_ATTACK_TIMER_GUARD_FRAMES, Z_BUTTON_TIMER_OFFSET,
        patch_shoulder_timers,
    };
    use crate::core::input::ControllerState;

    #[test]
    fn held_left_shoulder_patches_both_slide_timers() {
        let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
        let controller_state =
            ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_L2), 0.0, 0.0, 0.0, 0.0);

        patch_shoulder_timers(&mut memory, controller_state).unwrap();

        assert_eq!(
            read_i16(&memory, Z_BUTTON_TIMER_OFFSET),
            SLIDE_ATTACK_TIMER_GUARD_FRAMES,
        );
        assert_eq!(
            read_i16(&memory, R_BUTTON_TIMER_OFFSET),
            SLIDE_ATTACK_TIMER_GUARD_FRAMES,
        );
    }

    #[test]
    fn held_right_shoulder_patches_both_slide_timers() {
        let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
        let controller_state =
            ControllerState::from_normalized(button_mask(DEVICE_ID_JOYPAD_R), 0.0, 0.0, 0.0, 0.0);

        patch_shoulder_timers(&mut memory, controller_state).unwrap();

        assert_eq!(
            read_i16(&memory, Z_BUTTON_TIMER_OFFSET),
            SLIDE_ATTACK_TIMER_GUARD_FRAMES,
        );
        assert_eq!(
            read_i16(&memory, R_BUTTON_TIMER_OFFSET),
            SLIDE_ATTACK_TIMER_GUARD_FRAMES,
        );
    }

    #[test]
    fn idle_shoulders_leave_timers_unchanged() {
        let mut memory = vec![0_u8; R_BUTTON_TIMER_OFFSET + 2];
        let controller_state = ControllerState::default();

        patch_shoulder_timers(&mut memory, controller_state).unwrap();

        assert_eq!(read_i16(&memory, Z_BUTTON_TIMER_OFFSET), 0);
        assert_eq!(read_i16(&memory, R_BUTTON_TIMER_OFFSET), 0);
    }

    fn button_mask(button_id: u32) -> u16 {
        1_u16 << button_id
    }

    fn read_i16(memory: &[u8], offset: usize) -> i16 {
        i16::from_le_bytes([memory[offset], memory[offset + 1]])
    }
}
