// rust/bindings/emulator.rs
//! Python binding for the native libretro host runtime.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::bindings::error::map_core_error;
use crate::core::host::{Host, RepeatedStepConfig};
use crate::core::input::ControllerState;
use crate::core::observation::ObservationPreset;

mod frame;
mod state;
mod step;
mod telemetry;

use frame::frame_to_pyarray;
pub use state::encode_state_flags;
pub use step::{PyStepStatus, PyStepSummary};
use step::{step_status_to_py, step_summary_to_py};
use telemetry::telemetry_to_py;
pub use telemetry::{PyPlayerTelemetry, PyTelemetry};

/// Python-facing wrapper around one native `Host` instance.
#[pyclass(name = "Emulator", unsendable)]
pub struct PyEmulator {
    host: Host,
}

#[pymethods]
impl PyEmulator {
    #[new]
    #[pyo3(signature = (core_path, rom_path, runtime_dir=None, baseline_state_path=None, renderer="angrylion"))]
    fn new(
        py: Python<'_>,
        core_path: &str,
        rom_path: &str,
        runtime_dir: Option<&str>,
        baseline_state_path: Option<&str>,
        renderer: &str,
    ) -> PyResult<Self> {
        let host = py
            .detach(|| {
                Host::open(
                    Path::new(core_path),
                    Path::new(rom_path),
                    runtime_dir.map(Path::new),
                    baseline_state_path.map(Path::new),
                    renderer,
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

    #[pyo3(signature = (count=1, capture_video=true))]
    fn step_frames(&mut self, py: Python<'_>, count: usize, capture_video: bool) -> PyResult<()> {
        py.detach(|| self.host.step_frames(count, capture_video))
            .map_err(map_core_error)
    }

    #[pyo3(signature = (
        action_repeat,
        preset,
        frame_stack,
        stuck_min_speed_kph,
        energy_loss_epsilon,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_timer_limit,
        joypad_mask=0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn step_repeat_raw<'py>(
        &mut self,
        py: Python<'py>,
        action_repeat: usize,
        preset: &str,
        frame_stack: usize,
        stuck_min_speed_kph: f32,
        energy_loss_epsilon: f32,
        max_episode_steps: usize,
        stuck_step_limit: usize,
        wrong_way_timer_limit: usize,
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
        let spec = py
            .detach(|| self.host.observation_spec(preset))
            .map_err(map_core_error)?;
        let controller_state = ControllerState::from_normalized(
            joypad_mask,
            left_stick_x,
            left_stick_y,
            right_stick_x,
            right_stick_y,
        );
        let result = py
            .detach(|| {
                self.host.step_repeat_raw(RepeatedStepConfig {
                    controller_state,
                    action_repeat,
                    preset,
                    frame_stack,
                    stuck_min_speed_kph,
                    energy_loss_epsilon,
                    max_episode_steps,
                    stuck_step_limit,
                    wrong_way_timer_limit,
                })
            })
            .map_err(map_core_error)?;
        let observation = frame_to_pyarray(
            py,
            result.observation,
            spec.frame_height,
            spec.frame_width,
            spec.channels * frame_stack,
        )?;
        let summary = step_summary_to_py(py, &result.summary)?;
        let status = step_status_to_py(py, &result.status)?;
        let telemetry = telemetry_to_py(py, &result.final_telemetry)?;
        PyTuple::new(
            py,
            [
                observation,
                summary.into_bound(py).into_any(),
                status.into_bound(py).into_any(),
                telemetry.into_bound(py).into_any(),
            ],
        )
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

    fn frame_rgb<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let frame = self.host.frame_rgb().map_err(map_core_error)?;
        Ok(PyBytes::new(py, frame))
    }

    fn observation_spec<'py>(
        &mut self,
        py: Python<'py>,
        preset: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
        let spec = py
            .detach(|| self.host.observation_spec(preset))
            .map_err(map_core_error)?;
        let dict = PyDict::new(py);
        dict.set_item("preset", spec.preset_name)?;
        dict.set_item("width", spec.frame_width)?;
        dict.set_item("height", spec.frame_height)?;
        dict.set_item("channels", spec.channels)?;
        dict.set_item("display_width", spec.display_width)?;
        dict.set_item("display_height", spec.display_height)?;
        Ok(dict)
    }

    #[pyo3(signature = (preset, frame_stack))]
    fn frame_observation<'py>(
        &mut self,
        py: Python<'py>,
        preset: &str,
        frame_stack: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
        let spec = py
            .detach(|| self.host.observation_spec(preset))
            .map_err(map_core_error)?;
        let frame = py
            .detach(|| self.host.observation_frame(preset, frame_stack))
            .map_err(map_core_error)?;
        frame_to_pyarray(
            py,
            frame,
            spec.frame_height,
            spec.frame_width,
            spec.channels * frame_stack,
        )
    }

    #[pyo3(signature = (preset))]
    fn frame_display<'py>(&mut self, py: Python<'py>, preset: &str) -> PyResult<Bound<'py, PyAny>> {
        let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
        let spec = py
            .detach(|| self.host.observation_spec(preset))
            .map_err(map_core_error)?;
        let frame = py
            .detach(|| self.host.display_frame(preset))
            .map_err(map_core_error)?;
        frame_to_pyarray(py, frame, spec.display_height, spec.display_width, 3)
    }

    fn telemetry(&mut self, py: Python<'_>) -> PyResult<Py<PyTelemetry>> {
        let telemetry = py
            .detach(|| self.host.telemetry())
            .map_err(map_core_error)?;
        telemetry_to_py(py, &telemetry)
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

    fn game_rng_state(&mut self, py: Python<'_>) -> PyResult<(u32, u32, u32, u32)> {
        let state = py
            .detach(|| self.host.game_rng_state())
            .map_err(map_core_error)?;
        Ok(state.as_tuple())
    }

    fn randomize_game_rng(&mut self, py: Python<'_>, seed: u64) -> PyResult<(u32, u32, u32, u32)> {
        let state = py
            .detach(|| self.host.randomize_game_rng(seed))
            .map_err(map_core_error)?;
        Ok(state.as_tuple())
    }

    fn close(&mut self) {
        self.host.close();
    }
}
