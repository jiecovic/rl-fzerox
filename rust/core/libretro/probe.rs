// rust/core/libretro/probe.rs
//! Lightweight metadata probe for libretro cores without constructing the full
//! higher-level emulator host runtime.
//!
//! This module is for loader/core diagnostics only. It verifies that the shared
//! library can be opened and that the libretro metadata entry points can be
//! queried. It intentionally does not load content, allocate an emulator
//! runtime, or inspect game RAM/save-RAM.

use std::path::Path;

use crate::core::api::LoadedCore;
use crate::core::error::CoreError;
use crate::core::info::CoreInfo;

/// Load just enough of a core to inspect its advertised metadata.
pub fn probe(core_path: &Path) -> Result<CoreInfo, CoreError> {
    Ok(LoadedCore::load(core_path)?.info().clone())
}
