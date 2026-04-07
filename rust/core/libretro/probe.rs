// rust/core/libretro/probe.rs
use std::path::Path;

use crate::core::api::LoadedCore;
use crate::core::error::CoreError;
use crate::core::info::CoreInfo;

pub fn probe(core_path: &Path) -> Result<CoreInfo, CoreError> {
    Ok(LoadedCore::load(core_path)?.info().clone())
}
