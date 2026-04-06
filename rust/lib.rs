// rust/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod bindings;
mod core;

use crate::bindings::{PyCoreInfo, PyEmulator, probe_core};

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEmulator>()?;
    module.add_class::<PyCoreInfo>()?;
    module.add_function(wrap_pyfunction!(probe_core, module)?)?;
    Ok(())
}
