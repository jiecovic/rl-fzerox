// rust/bindings/emulator/methods/control.rs
//! Runtime-control, state, RAM, RNG, and telemetry method bodies.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::bindings::emulator::telemetry::telemetry_to_py;
use crate::bindings::emulator::{PyEmulator, PyTelemetry};
use crate::bindings::error::map_core_error;
use crate::core::input::ControllerState;

pub(in crate::bindings::emulator) fn reset(
    emulator: &mut PyEmulator,
    py: Python<'_>,
) -> PyResult<()> {
    py.detach(|| emulator.host.reset()).map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn step_frames(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    count: usize,
    capture_video: bool,
) -> PyResult<()> {
    py.detach(|| emulator.host.step_frames(count, capture_video))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn set_controller_state(
    emulator: &mut PyEmulator,
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
    py.detach(|| emulator.host.set_controller_state(controller_state))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn save_state(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    path: &str,
) -> PyResult<()> {
    py.detach(|| emulator.host.save_state(Path::new(path)))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn load_baseline(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    path: &str,
) -> PyResult<()> {
    py.detach(|| emulator.host.load_baseline(Path::new(path)))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn load_baseline_bytes(
    emulator: &mut PyEmulator,
    state: &Bound<'_, PyBytes>,
) -> PyResult<()> {
    emulator
        .host
        .load_baseline_bytes(state.as_bytes())
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn capture_current_as_baseline(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    path: Option<&str>,
) -> PyResult<()> {
    py.detach(|| {
        emulator
            .host
            .capture_current_as_baseline(path.map(Path::new))
    })
    .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn telemetry(
    emulator: &mut PyEmulator,
    py: Python<'_>,
) -> PyResult<Py<PyTelemetry>> {
    let telemetry = py
        .detach(|| emulator.host.telemetry())
        .map_err(map_core_error)?;
    telemetry_to_py(py, &telemetry)
}

pub(in crate::bindings::emulator) fn read_system_ram<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    offset: usize,
    length: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = py
        .detach(|| emulator.host.read_system_ram(offset, length))
        .map_err(map_core_error)?;
    Ok(PyBytes::new(py, &bytes))
}

pub(in crate::bindings::emulator) fn write_system_ram(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    offset: usize,
    data: &Bound<'_, PyBytes>,
) -> PyResult<()> {
    let bytes = data.as_bytes().to_vec();
    py.detach(|| emulator.host.write_system_ram(offset, &bytes))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn game_rng_state(
    emulator: &mut PyEmulator,
    py: Python<'_>,
) -> PyResult<(u32, u32, u32, u32)> {
    let state = py
        .detach(|| emulator.host.game_rng_state())
        .map_err(map_core_error)?;
    Ok(state.as_tuple())
}

pub(in crate::bindings::emulator) fn randomize_game_rng(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    seed: u64,
) -> PyResult<(u32, u32, u32, u32)> {
    let state = py
        .detach(|| emulator.host.randomize_game_rng(seed))
        .map_err(map_core_error)?;
    Ok(state.as_tuple())
}

pub(in crate::bindings::emulator) fn close(emulator: &mut PyEmulator) {
    emulator.host.close();
}
