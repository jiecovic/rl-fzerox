// rust/bindings/emulator.rs
use std::path::Path;

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::error::CoreError;
use crate::core::host::Host;

#[pyclass(name = "Emulator", unsendable)]
pub struct PyEmulator {
    host: Host,
}

#[pymethods]
impl PyEmulator {
    #[new]
    fn new(core_path: &str, rom_path: &str) -> PyResult<Self> {
        let host = Host::open(Path::new(core_path), Path::new(rom_path)).map_err(map_core_error)?;
        Ok(Self { host })
    }

    #[getter]
    fn name(&self) -> String {
        self.host.name().to_owned()
    }

    #[getter]
    fn native_fps(&self) -> f64 {
        self.host.native_fps()
    }

    #[getter]
    fn display_aspect_ratio(&self) -> f64 {
        self.host.display_aspect_ratio()
    }

    #[getter]
    fn frame_shape(&self) -> (usize, usize, usize) {
        self.host.frame_shape()
    }

    #[getter]
    fn frame_index(&self) -> usize {
        self.host.frame_index()
    }

    fn reset(&mut self) -> PyResult<()> {
        self.host.reset().map_err(map_core_error)
    }

    #[pyo3(signature = (count=1))]
    fn step_frames(&mut self, count: usize) -> PyResult<()> {
        self.host.step_frames(count).map_err(map_core_error)
    }

    fn frame_rgb<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let frame = self.host.frame_rgb().map_err(map_core_error)?;
        Ok(PyBytes::new(py, &frame))
    }

    fn close(&mut self) {
        self.host.close();
    }
}

fn map_core_error(error: CoreError) -> PyErr {
    match error {
        CoreError::MissingCore(_) | CoreError::MissingRom(_) => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
