// rust/core/host/runtime/lean_slide.rs
//! RAM-assisted lean slide handling for F-Zero X.
//!
//! F-Zero X uses a short Z/R timer window to detect double-tap side attacks.
//! When the policy requests a slide, keep normal controller input semantics but
//! patch the slide timers past that window before the frame update. That lets
//! the policy release early without the next lean edge turning into a dash.

use std::mem::size_of;

use libretro_sys::{DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R, MEMORY_SYSTEM_RAM};

use super::host::Host;
use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::telemetry::{player_r_button_timer_offset, player_z_button_timer_offset};

const Z_BUTTON_TIMER_OFFSET: usize = player_z_button_timer_offset();
const R_BUTTON_TIMER_OFFSET: usize = player_r_button_timer_offset();

#[derive(Clone, Copy, Debug)]
struct SlideAssistRam {
    lean_button_ids: [u32; 2],
    timer_offsets: [usize; 2],
    attack_timer_guard_frames: i16,
}

const SLIDE_ASSIST_RAM: SlideAssistRam = SlideAssistRam {
    lean_button_ids: [DEVICE_ID_JOYPAD_L2, DEVICE_ID_JOYPAD_R],
    timer_offsets: [Z_BUTTON_TIMER_OFFSET, R_BUTTON_TIMER_OFFSET],
    attack_timer_guard_frames: 15,
};

impl Host {
    pub(super) fn patch_lean_timers_for_slide_assist(
        &mut self,
        controller_state: ControllerState,
    ) -> Result<(), CoreError> {
        if !has_held_slide_lean(controller_state) {
            return Ok(());
        }
        let system_ram = self.system_ram_slice_mut()?;
        patch_lean_timers(system_ram, controller_state)
    }
}

fn has_held_slide_lean(controller_state: ControllerState) -> bool {
    SLIDE_ASSIST_RAM
        .lean_button_ids
        .iter()
        .any(|button_id| controller_state.joypad_state(*button_id) != 0)
}

fn patch_lean_timers(
    system_ram: &mut [u8],
    controller_state: ControllerState,
) -> Result<(), CoreError> {
    if !has_held_slide_lean(controller_state) {
        return Ok(());
    }

    // The two timers are adjacent `s16` fields. Guard both together so the
    // slide primitive remains safe even on byte/halfword-swapped RDRAM views.
    for offset in SLIDE_ASSIST_RAM.timer_offsets {
        write_i16(
            system_ram,
            offset,
            SLIDE_ASSIST_RAM.attack_timer_guard_frames,
        )?;
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
mod tests;
