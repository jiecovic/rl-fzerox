// rust/bindings/rom.rs
//! Python bindings for lightweight ROM identity validation.

use std::path::Path;

use pyo3::prelude::*;

use crate::bindings::error::map_core_error;
use crate::core::error::CoreError;
use crate::core::rom::validate_supported_rom;

/// Read a ROM file and verify it is the supported US F-Zero X revision.
///
/// This uses the same fixed-offset ROM identity check as `Host::open`, but it
/// does not load a libretro core, boot content, allocate video, or touch RAM.
#[pyfunction]
pub fn validate_supported_rom_path(rom_path: &str) -> PyResult<()> {
    let path = Path::new(rom_path);
    if !path.is_file() {
        return Err(map_core_error(CoreError::MissingRom(path.to_path_buf())));
    }
    let rom_bytes = std::fs::read(path).map_err(|error| {
        map_core_error(CoreError::ReadFile {
            path: path.to_path_buf(),
            source: error,
        })
    })?;
    validate_supported_rom(path, &rom_bytes).map_err(map_core_error)
}
