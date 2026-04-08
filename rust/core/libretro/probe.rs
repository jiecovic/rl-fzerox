// rust/core/libretro/probe.rs
//! Lightweight metadata probe for libretro cores without constructing the full
//! higher-level emulator host runtime.

use std::path::Path;

use crate::core::api::LoadedCore;
use crate::core::error::CoreError;
use crate::core::info::CoreInfo;

/// Load just enough of a core to inspect its advertised metadata.
pub fn probe(core_path: &Path) -> Result<CoreInfo, CoreError> {
    Ok(LoadedCore::load(core_path)?.info().clone())
}
