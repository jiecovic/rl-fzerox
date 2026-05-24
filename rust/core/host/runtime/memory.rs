// rust/core/host/runtime/memory.rs
//! Raw libretro memory access helpers used by telemetry extraction and debug
//! tooling.

use std::slice;

use super::host::Host;
use crate::core::error::CoreError;

impl Host {
    pub fn system_ram_size(&mut self) -> Result<usize, CoreError> {
        self.ensure_open()?;
        Ok(self.system_ram_size)
    }

    pub fn read_system_ram(&mut self, offset: usize, length: usize) -> Result<Vec<u8>, CoreError> {
        self.read_memory(libretro_sys::MEMORY_SYSTEM_RAM, offset, length)
    }

    pub fn write_system_ram(&mut self, offset: usize, bytes: &[u8]) -> Result<(), CoreError> {
        self.write_memory(libretro_sys::MEMORY_SYSTEM_RAM, offset, bytes)
    }

    pub(super) fn memory_size(&mut self, memory_id: u32) -> Result<usize, CoreError> {
        self.ensure_open()?;
        let size = self.call_core(|core| unsafe { core.memory_size(memory_id) });
        if size == 0 {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }
        Ok(size)
    }

    pub(super) fn cache_system_ram_size(&mut self) -> Result<(), CoreError> {
        self.system_ram_size = self.memory_size(libretro_sys::MEMORY_SYSTEM_RAM)?;
        Ok(())
    }

    pub(super) fn system_ram_slice(&mut self) -> Result<&[u8], CoreError> {
        self.ensure_open()?;
        let available = self.available_memory_size(libretro_sys::MEMORY_SYSTEM_RAM)?;
        let data =
            self.call_core(|core| unsafe { core.memory_data(libretro_sys::MEMORY_SYSTEM_RAM) });
        if data.is_null() {
            return Err(CoreError::MemoryUnavailable {
                memory_id: libretro_sys::MEMORY_SYSTEM_RAM,
            });
        }
        let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>(), available) };
        Ok(bytes)
    }

    pub(super) fn system_ram_slice_mut(&mut self) -> Result<&mut [u8], CoreError> {
        self.ensure_open()?;
        let available = self.available_memory_size(libretro_sys::MEMORY_SYSTEM_RAM)?;
        let data =
            self.call_core(|core| unsafe { core.memory_data(libretro_sys::MEMORY_SYSTEM_RAM) });
        if data.is_null() {
            return Err(CoreError::MemoryUnavailable {
                memory_id: libretro_sys::MEMORY_SYSTEM_RAM,
            });
        }
        let bytes = unsafe { slice::from_raw_parts_mut(data.cast::<u8>(), available) };
        Ok(bytes)
    }

    pub(super) fn read_memory(
        &mut self,
        memory_id: u32,
        offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, CoreError> {
        // The public Python API expects an owned byte buffer here, so this
        // helper intentionally copies out the requested subrange.
        self.ensure_open()?;
        let available = self.available_memory_size(memory_id)?;
        let end = offset
            .checked_add(length)
            .ok_or(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            })?;
        if end > available {
            return Err(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            });
        }

        let data = self.call_core(|core| unsafe { core.memory_data(memory_id) });
        if data.is_null() {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }

        let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>().add(offset), length) };
        Ok(bytes.to_vec())
    }

    pub(super) fn write_memory(
        &mut self,
        memory_id: u32,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CoreError> {
        self.ensure_open()?;
        let available = self.available_memory_size(memory_id)?;
        let length = bytes.len();
        let end = offset
            .checked_add(length)
            .ok_or(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            })?;
        if end > available {
            return Err(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            });
        }

        let data = self.call_core(|core| unsafe { core.memory_data(memory_id) });
        if data.is_null() {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }

        let target = unsafe { slice::from_raw_parts_mut(data.cast::<u8>().add(offset), length) };
        target.copy_from_slice(bytes);
        Ok(())
    }

    fn available_memory_size(&mut self, memory_id: u32) -> Result<usize, CoreError> {
        if memory_id == libretro_sys::MEMORY_SYSTEM_RAM && self.system_ram_size != 0 {
            return Ok(self.system_ram_size);
        }
        self.memory_size(memory_id)
    }
}
