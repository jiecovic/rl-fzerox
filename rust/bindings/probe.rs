// rust/bindings/probe.rs
//! Python binding for lightweight libretro core metadata probing.
//!
//! The probe path deliberately stops at dynamic loading and libretro system
//! metadata. It is useful when debugging a core path, native extension build, or
//! loader issue without requiring a ROM, runtime directory, save state, or full
//! emulator host construction.

use std::path::Path;

use pyo3::prelude::*;

use crate::bindings::error::map_core_error;
use crate::core::info::CoreInfo;
use crate::core::probe::probe;

/// Python-facing owned view of libretro core metadata.
#[pyclass(name = "CoreInfo", frozen, get_all, skip_from_py_object)]
pub struct PyCoreInfo {
    pub api_version: u32,
    pub library_name: String,
    pub library_version: String,
    pub valid_extensions: Vec<String>,
    pub requires_full_path: bool,
    pub blocks_extract: bool,
}

/// Load a libretro core and return its advertised metadata.
///
/// This does not initialize content, boot a game, read system RAM, or touch
/// save-RAM. It is intentionally narrower than constructing `Emulator`.
#[pyfunction]
pub fn probe_core(core_path: &str) -> PyResult<PyCoreInfo> {
    let core_info = probe(Path::new(core_path)).map_err(map_core_error)?;
    Ok(PyCoreInfo::from(core_info))
}

/// Convert the native metadata struct into the Python-facing equivalent.
impl From<CoreInfo> for PyCoreInfo {
    fn from(value: CoreInfo) -> Self {
        Self {
            api_version: value.api_version,
            library_name: value.library_name,
            library_version: value.library_version,
            valid_extensions: value.valid_extensions,
            requires_full_path: value.requires_full_path,
            blocks_extract: value.blocks_extract,
        }
    }
}
