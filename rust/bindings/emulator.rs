// rust/bindings/emulator.rs
//! Python binding for the native libretro host runtime.

use std::path::Path;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::bindings::error::map_core_error;
use crate::core::host::Host;
use crate::core::input::ControllerState;
use crate::core::observation::ObservationPreset;
use crate::core::telemetry::{PlayerTelemetry, TelemetrySnapshot};

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

    fn telemetry<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let telemetry = py
            .detach(|| self.host.telemetry())
            .map_err(map_core_error)?;
        telemetry_to_pydict(py, &telemetry)
    }

    fn telemetry_flat<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let telemetry = py
            .detach(|| self.host.telemetry())
            .map_err(map_core_error)?;
        telemetry_to_pytuple(py, &telemetry)
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

/// Convert the rich native telemetry snapshot into a Python dictionary for
/// introspection/debugging use.
fn telemetry_to_pydict<'py>(
    py: Python<'py>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("system_ram_size", telemetry.system_ram_size)?;
    dict.set_item("game_frame_count", telemetry.game_frame_count)?;
    dict.set_item("game_mode_raw", telemetry.game_mode_raw)?;
    dict.set_item("game_mode_name", &telemetry.game_mode_name)?;
    dict.set_item("course_index", telemetry.course_index)?;
    dict.set_item("in_race_mode", telemetry.in_race_mode)?;
    dict.set_item("player", player_to_pydict(py, &telemetry.player)?)?;
    Ok(dict)
}

/// Convert the player telemetry sub-struct into a Python dictionary.
fn player_to_pydict<'py>(
    py: Python<'py>,
    player: &PlayerTelemetry,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("state_flags", player.state_flags)?;
    dict.set_item("state_labels", PyList::new(py, &player.state_labels)?)?;
    dict.set_item("speed_raw", player.speed_raw)?;
    dict.set_item("speed_kph", player.speed_kph)?;
    dict.set_item("energy", player.energy)?;
    dict.set_item("max_energy", player.max_energy)?;
    dict.set_item("boost_timer", player.boost_timer)?;
    dict.set_item("race_distance", player.race_distance)?;
    dict.set_item("laps_completed_distance", player.laps_completed_distance)?;
    dict.set_item("lap_distance", player.lap_distance)?;
    dict.set_item("race_distance_position", player.race_distance_position)?;
    dict.set_item("race_time_ms", player.race_time_ms)?;
    dict.set_item("lap", player.lap)?;
    dict.set_item("laps_completed", player.laps_completed)?;
    dict.set_item("position", player.position)?;
    dict.set_item("character", player.character)?;
    dict.set_item("machine_index", player.machine_index)?;
    Ok(dict)
}

/// Convert the hot-path telemetry shape into tuples so Python can rebuild typed
/// telemetry objects with less object churn than nested dict/list creation.
fn telemetry_to_pytuple<'py>(
    py: Python<'py>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Bound<'py, PyTuple>> {
    let player = player_to_pytuple(py, &telemetry.player)?;
    PyTuple::new(
        py,
        [
            telemetry.system_ram_size.into_pyobject(py)?.into_any(),
            telemetry.game_frame_count.into_pyobject(py)?.into_any(),
            telemetry.game_mode_raw.into_pyobject(py)?.into_any(),
            telemetry
                .game_mode_name
                .as_str()
                .into_pyobject(py)?
                .into_any(),
            telemetry.course_index.into_pyobject(py)?.into_any(),
            telemetry
                .in_race_mode
                .into_pyobject(py)?
                .to_owned()
                .into_any(),
            player.into_any(),
        ],
    )
}

/// Convert the player telemetry sub-struct into the tuple layout expected by
/// the Python-side fast path.
fn player_to_pytuple<'py>(
    py: Python<'py>,
    player: &PlayerTelemetry,
) -> PyResult<Bound<'py, PyTuple>> {
    let state_labels = PyTuple::new(py, player.state_labels.iter().copied())?;
    PyTuple::new(
        py,
        [
            player.state_flags.into_pyobject(py)?.into_any(),
            state_labels.into_any(),
            player.speed_raw.into_pyobject(py)?.into_any(),
            player.speed_kph.into_pyobject(py)?.into_any(),
            player.energy.into_pyobject(py)?.into_any(),
            player.max_energy.into_pyobject(py)?.into_any(),
            player.boost_timer.into_pyobject(py)?.into_any(),
            player.race_distance.into_pyobject(py)?.into_any(),
            player.laps_completed_distance.into_pyobject(py)?.into_any(),
            player.lap_distance.into_pyobject(py)?.into_any(),
            player.race_distance_position.into_pyobject(py)?.into_any(),
            player.race_time_ms.into_pyobject(py)?.into_any(),
            player.lap.into_pyobject(py)?.into_any(),
            player.laps_completed.into_pyobject(py)?.into_any(),
            player.position.into_pyobject(py)?.into_any(),
            player.character.into_pyobject(py)?.into_any(),
            player.machine_index.into_pyobject(py)?.into_any(),
        ],
    )
}

/// Materialize a Python-owned NumPy array view of a frame buffer.
///
/// This still performs one copy into Python-owned memory; it just avoids the
/// extra `PyBytes` hop that the older path used.
fn frame_to_pyarray<'py>(
    py: Python<'py>,
    frame: &[u8],
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let array = PyArray1::<u8>::from_vec(py, frame.to_vec());
    Ok(array.reshape([height, width, channels])?.into_any())
}
