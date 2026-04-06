// rust/bindings/probe.rs
use std::path::Path;

use pyo3::prelude::*;

use crate::bindings::error::map_core_error;
use crate::core::info::CoreInfo;
use crate::core::probe::probe;

#[pyclass(name = "CoreInfo", frozen, get_all, skip_from_py_object)]
pub struct PyCoreInfo {
    pub api_version: u32,
    pub library_name: String,
    pub library_version: String,
    pub valid_extensions: Vec<String>,
    pub requires_full_path: bool,
    pub blocks_extract: bool,
}

#[pyfunction]
pub fn probe_core(core_path: &str) -> PyResult<PyCoreInfo> {
    let core_info = probe(Path::new(core_path)).map_err(map_core_error)?;
    Ok(PyCoreInfo::from(core_info))
}

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
