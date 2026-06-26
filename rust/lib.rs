// rust/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod bindings;
mod core;

use crate::bindings::{
    PyCoreInfo, PyEmulator, PyPlayerTelemetry, PyStepStatus, PyStepSummary, PyTelemetry,
    display_size, encode_state_flags, probe_core, register_input_api, stacked_observation_channels,
    validate_supported_rom_path,
};

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEmulator>()?;
    module.add_class::<PyCoreInfo>()?;
    module.add_class::<PyPlayerTelemetry>()?;
    module.add_class::<PyTelemetry>()?;
    module.add_class::<PyStepSummary>()?;
    module.add_class::<PyStepStatus>()?;
    module.add_function(wrap_pyfunction!(encode_state_flags, module)?)?;
    module.add_function(wrap_pyfunction!(probe_core, module)?)?;
    module.add_function(wrap_pyfunction!(display_size, module)?)?;
    module.add_function(wrap_pyfunction!(stacked_observation_channels, module)?)?;
    module.add_function(wrap_pyfunction!(validate_supported_rom_path, module)?)?;
    register_input_api(module)?;
    Ok(())
}
