// rust/bindings/emulator/telemetry.rs
//! PyO3 telemetry objects exposed directly to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bindings::emulator::state::{
    FLAG_ACTIVE, FLAG_AIRBORNE, FLAG_CAN_BOOST, FLAG_COLLISION_RECOIL, FLAG_CPU_CONTROLLED,
    FLAG_CRASHED, FLAG_DASH_PAD_BOOST, FLAG_FALLING_OFF_TRACK, FLAG_FINISHED, FLAG_RETIRED,
    FLAG_SPINNING_OUT, has_state_flag, state_flag_labels,
};
use crate::core::telemetry::{PlayerTelemetry, TelemetrySnapshot};

#[pyclass(
    name = "PlayerTelemetry",
    module = "fzerox_emulator._native",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub struct PyPlayerTelemetry {
    inner: PlayerTelemetry,
}

#[pymethods]
impl PyPlayerTelemetry {
    #[new]
    #[pyo3(signature = (
        state_flags,
        speed_kph,
        energy,
        max_energy,
        boost_timer,
        reverse_timer,
        race_distance,
        lap_distance,
        race_time_ms,
        lap,
        laps_completed,
        position,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_flags: u32,
        speed_kph: f32,
        energy: f32,
        max_energy: f32,
        boost_timer: i32,
        reverse_timer: i32,
        race_distance: f32,
        lap_distance: f32,
        race_time_ms: i32,
        lap: i16,
        laps_completed: i16,
        position: i32,
    ) -> Self {
        Self {
            inner: PlayerTelemetry {
                state_flags,
                speed_kph,
                energy,
                max_energy,
                boost_timer,
                reverse_timer,
                race_distance,
                lap_distance,
                race_time_ms,
                lap,
                laps_completed,
                position,
            },
        }
    }

    #[getter]
    fn state_flags(&self) -> u32 {
        self.inner.state_flags
    }

    #[getter]
    fn state_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, state_flag_labels(self.inner.state_flags))
    }

    #[getter]
    fn speed_kph(&self) -> f32 {
        self.inner.speed_kph
    }

    #[getter]
    fn energy(&self) -> f32 {
        self.inner.energy
    }

    #[getter]
    fn max_energy(&self) -> f32 {
        self.inner.max_energy
    }

    #[getter]
    fn boost_timer(&self) -> i32 {
        self.inner.boost_timer
    }

    #[getter]
    fn reverse_timer(&self) -> i32 {
        self.inner.reverse_timer
    }

    #[getter]
    fn race_distance(&self) -> f32 {
        self.inner.race_distance
    }

    #[getter]
    fn lap_distance(&self) -> f32 {
        self.inner.lap_distance
    }

    #[getter]
    fn race_time_ms(&self) -> i32 {
        self.inner.race_time_ms
    }

    #[getter]
    fn lap(&self) -> i16 {
        self.inner.lap
    }

    #[getter]
    fn laps_completed(&self) -> i16 {
        self.inner.laps_completed
    }

    #[getter]
    fn position(&self) -> i32 {
        self.inner.position
    }

    #[getter]
    fn collision_recoil(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_COLLISION_RECOIL)
    }

    #[getter]
    fn spinning_out(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_SPINNING_OUT)
    }

    #[getter]
    fn retired(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_RETIRED)
    }

    #[getter]
    fn falling_off_track(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_FALLING_OFF_TRACK)
    }

    #[getter]
    fn can_boost(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_CAN_BOOST)
    }

    #[getter]
    fn cpu_controlled(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_CPU_CONTROLLED)
    }

    #[getter]
    fn dash_pad_boost(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_DASH_PAD_BOOST)
    }

    #[getter]
    fn finished(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_FINISHED)
    }

    #[getter]
    fn airborne(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_AIRBORNE)
    }

    #[getter]
    fn crashed(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_CRASHED)
    }

    #[getter]
    fn active(&self) -> bool {
        has_state_flag(self.inner.state_flags, FLAG_ACTIVE)
    }

    #[getter]
    fn terminal_reason(&self) -> Option<&'static str> {
        self.inner.terminal_reason()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("state_flags", self.state_flags())?;
        dict.set_item("state_labels", self.state_labels(py)?)?;
        dict.set_item("speed_kph", self.speed_kph())?;
        dict.set_item("energy", self.energy())?;
        dict.set_item("max_energy", self.max_energy())?;
        dict.set_item("boost_timer", self.boost_timer())?;
        dict.set_item("reverse_timer", self.reverse_timer())?;
        dict.set_item("race_distance", self.race_distance())?;
        dict.set_item("lap_distance", self.lap_distance())?;
        dict.set_item("race_time_ms", self.race_time_ms())?;
        dict.set_item("lap", self.lap())?;
        dict.set_item("laps_completed", self.laps_completed())?;
        dict.set_item("position", self.position())?;
        Ok(dict)
    }
}

impl PyPlayerTelemetry {
    pub(super) fn from_native(player: &PlayerTelemetry) -> Self {
        Self {
            inner: player.clone(),
        }
    }
}

#[pyclass(
    name = "FZeroXTelemetry",
    module = "fzerox_emulator._native",
    frozen,
    skip_from_py_object
)]
pub struct PyTelemetry {
    total_lap_count: i32,
    game_mode_raw: u32,
    game_mode_name: String,
    in_race_mode: bool,
    total_racers: i32,
    course_index: u32,
    player: Py<PyPlayerTelemetry>,
}

#[pymethods]
impl PyTelemetry {
    #[new]
    #[pyo3(signature = (
        total_lap_count,
        game_mode_raw,
        game_mode_name,
        in_race_mode,
        total_racers,
        course_index,
        player,
    ))]
    fn new(
        total_lap_count: i32,
        game_mode_raw: u32,
        game_mode_name: String,
        in_race_mode: bool,
        total_racers: i32,
        course_index: u32,
        player: Py<PyPlayerTelemetry>,
    ) -> Self {
        Self {
            total_lap_count,
            game_mode_raw,
            game_mode_name,
            in_race_mode,
            total_racers,
            course_index,
            player,
        }
    }

    #[getter]
    fn total_lap_count(&self) -> i32 {
        self.total_lap_count
    }

    #[getter]
    fn game_mode_raw(&self) -> u32 {
        self.game_mode_raw
    }

    #[getter]
    fn game_mode_name(&self) -> &str {
        &self.game_mode_name
    }

    #[getter]
    fn in_race_mode(&self) -> bool {
        self.in_race_mode
    }

    #[getter]
    fn total_racers(&self) -> i32 {
        self.total_racers
    }

    #[getter]
    fn course_index(&self) -> u32 {
        self.course_index
    }

    #[getter]
    fn player(&self, py: Python<'_>) -> Py<PyPlayerTelemetry> {
        self.player.clone_ref(py)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_lap_count", self.total_lap_count())?;
        dict.set_item("game_mode_raw", self.game_mode_raw())?;
        dict.set_item("game_mode_name", self.game_mode_name())?;
        dict.set_item("in_race_mode", self.in_race_mode())?;
        dict.set_item("total_racers", self.total_racers())?;
        dict.set_item("course_index", self.course_index())?;
        let player_handle = self.player(py);
        let player = player_handle.bind(py);
        dict.set_item("player", player.call_method0("to_dict")?)?;
        Ok(dict)
    }
}

pub(super) fn telemetry_to_py(
    py: Python<'_>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Py<PyTelemetry>> {
    let player = Py::new(py, PyPlayerTelemetry::from_native(&telemetry.player))?;
    Py::new(
        py,
        PyTelemetry {
            total_lap_count: telemetry.total_lap_count,
            game_mode_raw: telemetry.game_mode_raw,
            game_mode_name: telemetry.game_mode_name.to_owned(),
            in_race_mode: telemetry.in_race_mode,
            total_racers: telemetry.total_racers,
            course_index: telemetry.course_index,
            player,
        },
    )
}
