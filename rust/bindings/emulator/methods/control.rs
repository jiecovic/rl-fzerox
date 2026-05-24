// rust/bindings/emulator/methods/control.rs
//! Runtime-control, state, RAM, RNG, and telemetry method bodies.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::bindings::emulator::telemetry::telemetry_to_py;
use crate::bindings::emulator::{PyEmulator, PyTelemetry};
use crate::bindings::error::map_core_error;
use crate::bindings::payload::{optional_item, required_item};
use crate::core::game::race_start::{RaceStartMode, RaceStartSetup};
use crate::core::input::ControllerState;

const RACE_START_PAYLOAD: &str = "race-start request";

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

pub(in crate::bindings::emulator) fn patch_race_start_setup(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let request = RaceStartBindingRequest::from_py_dict(request)?;
    py.detach(|| {
        emulator
            .host
            .patch_race_start_setup(request.mode, request.setup)
    })
    .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn patch_machine_settings(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let request = RaceStartBindingRequest::from_py_dict(request)?;
    py.detach(|| {
        emulator
            .host
            .patch_machine_settings(request.mode, request.setup)
    })
    .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn validate_race_start_setup(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let request = RaceStartBindingRequest::from_py_dict(request)?;
    py.detach(|| {
        emulator
            .host
            .validate_race_start_setup(request.mode, request.setup)
    })
    .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn patch_engine_settings(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    mode: &str,
    engine_setting_raw_value: i32,
) -> PyResult<()> {
    let mode = parse_race_start_mode(mode)?;
    py.detach(|| {
        emulator
            .host
            .patch_engine_settings(mode, engine_setting_raw_value)
    })
    .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn force_race_reinit(
    emulator: &mut PyEmulator,
    py: Python<'_>,
    mode: &str,
) -> PyResult<()> {
    let mode = parse_race_start_mode(mode)?;
    py.detach(|| emulator.host.force_race_reinit(mode))
        .map_err(map_core_error)
}

pub(in crate::bindings::emulator) fn patch_time_attack_menu_mode(
    emulator: &mut PyEmulator,
    py: Python<'_>,
) -> PyResult<()> {
    py.detach(|| emulator.host.patch_time_attack_menu_mode())
        .map_err(map_core_error)
}

fn parse_race_start_mode(mode: &str) -> PyResult<RaceStartMode> {
    RaceStartMode::parse(mode).map_err(map_core_error)
}

struct RaceStartBindingRequest {
    mode: RaceStartMode,
    setup: RaceStartSetup,
}

impl RaceStartBindingRequest {
    fn from_py_dict(request: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mode_raw: String = required_item(request, RACE_START_PAYLOAD, "mode")?.extract()?;
        let mode = parse_race_start_mode(&mode_raw)?;
        let setup = RaceStartSetup {
            course_index: required_item(request, RACE_START_PAYLOAD, "course_index")?.extract()?,
            character_index: required_item(request, RACE_START_PAYLOAD, "character_index")?
                .extract()?,
            machine_skin_index: optional_item(request, "machine_skin_index", -1)?,
            engine_setting_raw_value: required_item(
                request,
                RACE_START_PAYLOAD,
                "engine_setting_raw_value",
            )?
            .extract()?,
            total_lap_count: optional_item(request, "total_lap_count", 3)?,
            gp_difficulty_raw_value: optional_item(request, "gp_difficulty_raw_value", -1)?,
        };
        Ok(Self { mode, setup })
    }
}

pub(in crate::bindings::emulator) fn vehicle_setup_info<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let setup = py
        .detach(|| emulator.host.vehicle_setup_info())
        .map_err(map_core_error)?;
    let dict = PyDict::new(py);
    dict.set_item("player_character_index_ram", setup.player_character_index)?;
    dict.set_item("racer_character_index_ram", setup.racer_character_index)?;
    dict.set_item("engine_setting_ram", setup.engine_setting)?;
    dict.set_item("engine_setting_percent_ram", setup.engine_setting * 100.0)?;
    dict.set_item(
        "character_engine_setting_ram",
        setup.character_engine_setting,
    )?;
    dict.set_item("racer_engine_curve_ram", setup.racer_engine_curve)?;
    Ok(dict)
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
