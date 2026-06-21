// rust/core/game/telemetry/read/scalars.rs
//! Scalar and pointer helpers shared by telemetry RAM readers.

use std::mem::size_of;

use crate::core::error::CoreError;
use crate::core::game::memory::read_f32;
use crate::core::telemetry::layout::{
    CAMERA, CameraRaceSetting, GLOBALS, GameMode, RaceDifficulty, TELEMETRY_CONFIG,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Vec3 {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) z: f32,
}

pub(super) fn player_reverse_timer_offset() -> usize {
    GLOBALS.reverse_timers + (TELEMETRY_CONFIG.player_racer_index * size_of::<i32>())
}

pub(super) fn player_damage_rumble_counter_offset() -> usize {
    GLOBALS.damage_rumble_counters + (TELEMETRY_CONFIG.player_racer_index * size_of::<i32>())
}

pub(super) fn player_camera_setting_offset() -> usize {
    GLOBALS.cameras + CAMERA.race_setting
}

pub(super) fn kseg0_pointer_to_offset(pointer: u32, memory_len: usize) -> Option<usize> {
    let address = pointer as usize;
    if address < TELEMETRY_CONFIG.kseg0_base {
        return None;
    }
    let offset = address - TELEMETRY_CONFIG.kseg0_base;
    (offset < memory_len).then_some(offset)
}

pub(super) fn resolve_game_mode(game_mode_raw: u32) -> Option<GameMode> {
    GameMode::try_from(game_mode_raw & 0x1F).ok()
}

pub(super) fn resolve_difficulty(difficulty_raw: i32) -> Option<RaceDifficulty> {
    RaceDifficulty::try_from(difficulty_raw).ok()
}

pub(super) fn resolve_camera_setting(camera_setting_raw: i32) -> Option<CameraRaceSetting> {
    CameraRaceSetting::try_from(camera_setting_raw).ok()
}

pub(super) fn read_vec3_magnitude(memory: &[u8], offset: usize) -> Result<f32, CoreError> {
    let x = read_f32(memory, offset)?;
    let y = read_f32(memory, offset + size_of::<f32>())?;
    let z = read_f32(memory, offset + (2 * size_of::<f32>()))?;
    Ok((x.mul_add(x, y.mul_add(y, z * z))).sqrt())
}

pub(super) fn read_vec3(memory: &[u8], offset: usize) -> Result<Vec3, CoreError> {
    Ok(Vec3 {
        x: read_f32(memory, offset)?,
        y: read_f32(memory, offset + size_of::<f32>())?,
        z: read_f32(memory, offset + (2 * size_of::<f32>()))?,
    })
}

pub(super) fn dot_vec3(
    memory: &[u8],
    lhs_offset: usize,
    rhs_offset: usize,
) -> Result<f32, CoreError> {
    let x = read_f32(memory, lhs_offset)? * read_f32(memory, rhs_offset)?;
    let y = read_f32(memory, lhs_offset + size_of::<f32>())?
        * read_f32(memory, rhs_offset + size_of::<f32>())?;
    let z = read_f32(memory, lhs_offset + (2 * size_of::<f32>()))?
        * read_f32(memory, rhs_offset + (2 * size_of::<f32>()))?;
    Ok(x + y + z)
}
