// rust/bindings/emulator.rs
use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::bindings::error::map_core_error;
use crate::core::host::Host;
use crate::core::input::ControllerState;

#[pyclass(name = "Emulator", unsendable)]
pub struct PyEmulator {
    host: Host,
}

#[pymethods]
impl PyEmulator {
    #[new]
    #[pyo3(signature = (core_path, rom_path, runtime_dir=None, baseline_state_path=None))]
    fn new(
        py: Python<'_>,
        core_path: &str,
        rom_path: &str,
        runtime_dir: Option<&str>,
        baseline_state_path: Option<&str>,
    ) -> PyResult<Self> {
        let host = py
            .detach(|| {
                Host::open(
                    Path::new(core_path),
                    Path::new(rom_path),
                    runtime_dir.map(Path::new),
                    baseline_state_path.map(Path::new),
                )
            })
            .map_err(map_core_error)?;
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

    #[getter]
    fn system_ram_size(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.host.system_ram_size())
            .map_err(map_core_error)
    }

    #[getter]
    fn baseline_kind(&self) -> &'static str {
        self.host.baseline_kind()
    }

    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.host.reset()).map_err(map_core_error)
    }

    #[pyo3(signature = (count=1))]
    fn step_frames(&mut self, py: Python<'_>, count: usize) -> PyResult<()> {
        py.detach(|| self.host.step_frames(count))
            .map_err(map_core_error)
    }

    #[pyo3(signature = (
        joypad_mask=0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
    ))]
    fn set_controller_state(
        &mut self,
        py: Python<'_>,
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> PyResult<()> {
        let controller_state = ControllerState::from_normalized(
            joypad_mask,
            left_stick_x,
            left_stick_y,
            right_stick_x,
            right_stick_y,
        );
        py.detach(|| self.host.set_controller_state(controller_state))
            .map_err(map_core_error)
    }

    fn save_state(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.host.save_state(Path::new(path)))
            .map_err(map_core_error)
    }

    #[pyo3(signature = (path=None))]
    fn capture_current_as_baseline(&mut self, py: Python<'_>, path: Option<&str>) -> PyResult<()> {
        py.detach(|| self.host.capture_current_as_baseline(path.map(Path::new)))
            .map_err(map_core_error)
    }

    fn frame_rgb<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let frame = self.host.frame_rgb().map_err(map_core_error)?;
        Ok(PyBytes::new(py, frame))
    }

    fn read_system_ram<'py>(
        &mut self,
        py: Python<'py>,
        offset: usize,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| self.host.read_system_ram(offset, length))
            .map_err(map_core_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn close(&mut self) {
        self.host.close();
    }
}
