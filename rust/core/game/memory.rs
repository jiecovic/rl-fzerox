//! Typed little-endian helpers for F-Zero X RDRAM layouts.

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;

pub(crate) fn read_i8(memory: &[u8], offset: usize) -> Result<i8, CoreError> {
    Ok(read_u8(memory, offset)? as i8)
}

pub(crate) fn read_i16(memory: &[u8], offset: usize) -> Result<i16, CoreError> {
    Ok(i16::from_le_bytes(read_array(memory, offset)?))
}

pub(crate) fn read_i32(memory: &[u8], offset: usize) -> Result<i32, CoreError> {
    Ok(i32::from_le_bytes(read_array(memory, offset)?))
}

pub(crate) fn read_u32(memory: &[u8], offset: usize) -> Result<u32, CoreError> {
    Ok(u32::from_le_bytes(read_array(memory, offset)?))
}

pub(crate) fn read_f32(memory: &[u8], offset: usize) -> Result<f32, CoreError> {
    Ok(f32::from_le_bytes(read_array(memory, offset)?))
}

pub(crate) fn write_i8(memory: &mut [u8], offset: usize, value: i8) -> Result<(), CoreError> {
    write_array(memory, offset, &value.to_le_bytes())
}

pub(crate) fn write_i16(memory: &mut [u8], offset: usize, value: i16) -> Result<(), CoreError> {
    write_array(memory, offset, &value.to_le_bytes())
}

pub(crate) fn write_i32(memory: &mut [u8], offset: usize, value: i32) -> Result<(), CoreError> {
    write_array(memory, offset, &value.to_le_bytes())
}

pub(crate) fn write_f32(memory: &mut [u8], offset: usize, value: f32) -> Result<(), CoreError> {
    write_array(memory, offset, &value.to_le_bytes())
}

pub(crate) fn read_word_swapped_u8(memory: &[u8], logical_offset: usize) -> Result<u8, CoreError> {
    read_u8(memory, logical_offset ^ 0x03)
}

fn read_u8(memory: &[u8], offset: usize) -> Result<u8, CoreError> {
    memory
        .get(offset)
        .copied()
        .ok_or_else(|| memory_error(offset, 1, memory.len()))
}

fn read_array<const N: usize>(memory: &[u8], offset: usize) -> Result<[u8; N], CoreError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| memory_error(offset, N, memory.len()))?;
    let bytes = memory
        .get(offset..end)
        .ok_or_else(|| memory_error(offset, N, memory.len()))?;
    let mut array = [0_u8; N];
    array.copy_from_slice(bytes);
    Ok(array)
}

fn write_array(memory: &mut [u8], offset: usize, bytes: &[u8]) -> Result<(), CoreError> {
    let length = bytes.len();
    let available = memory.len();
    let end = offset
        .checked_add(length)
        .ok_or_else(|| memory_error(offset, length, available))?;
    let target = memory
        .get_mut(offset..end)
        .ok_or_else(|| memory_error(offset, length, available))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn memory_error(offset: usize, length: usize, available: usize) -> CoreError {
    CoreError::MemoryOutOfRange {
        memory_id: MEMORY_SYSTEM_RAM,
        offset,
        length,
        available,
    }
}
