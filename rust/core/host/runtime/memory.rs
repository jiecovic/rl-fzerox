// rust/core/host/runtime/memory.rs
//! Raw libretro memory access helpers used by telemetry extraction and debug
//! tooling.

use std::slice;

use super::Host;
use crate::core::error::CoreError;

impl Host {
    pub(super) fn memory_size(&mut self, memory_id: u32) -> Result<usize, CoreError> {
        self.ensure_open()?;
        let size = self.call_core(|core| unsafe { core.memory_size(memory_id) });
        if size == 0 {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }
        Ok(size)
    }

    pub(super) fn system_ram_slice(&mut self) -> Result<&[u8], CoreError> {
        self.ensure_open()?;
        let available = self.memory_size(libretro_sys::MEMORY_SYSTEM_RAM)?;
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

    pub(super) fn read_memory(
        &mut self,
        memory_id: u32,
        offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, CoreError> {
        // The public Python API expects an owned byte buffer here, so this
        // helper intentionally copies out the requested subrange.
        self.ensure_open()?;
        let available = self.memory_size(memory_id)?;
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
}
