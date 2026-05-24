// rust/bindings/emulator/telemetry/root.rs
//! Python-facing root F-Zero X telemetry binding.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::bindings::emulator::telemetry::PyPlayerTelemetry;
use crate::core::telemetry::TelemetrySnapshot;

#[pyclass(
    name = "FZeroXTelemetry",
    module = "fzerox_emulator._native",
    frozen,
    skip_from_py_object
)]
pub struct PyTelemetry {
    total_lap_count: i32,
    difficulty_raw: i32,
    difficulty_name: String,
    camera_setting_raw: i32,
    camera_setting_name: String,
    race_intro_timer: i32,
    game_mode_raw: u32,
    game_mode_name: String,
    in_race_mode: bool,
    total_racers: i32,
    course_index: u32,
    course_segment_count: i32,
    course_length: f32,
    player: Py<PyPlayerTelemetry>,
}

#[pymethods]
impl PyTelemetry {
    #[new]
    #[pyo3(signature = (data))]
    fn new(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let difficulty_name: Option<String> = optional_item(data, "difficulty_name", None)?;
        let camera_setting_name: Option<String> = optional_item(data, "camera_setting_name", None)?;
        Ok(Self {
            total_lap_count: required_item(data, "total_lap_count")?.extract()?,
            difficulty_raw: optional_item(data, "difficulty_raw", 0)?,
            difficulty_name: difficulty_name.unwrap_or_else(|| "novice".to_owned()),
            camera_setting_raw: optional_item(data, "camera_setting_raw", 2)?,
            camera_setting_name: camera_setting_name.unwrap_or_else(|| "regular".to_owned()),
            race_intro_timer: optional_item(data, "race_intro_timer", 0)?,
            game_mode_raw: required_item(data, "game_mode_raw")?.extract()?,
            game_mode_name: required_item(data, "game_mode_name")?.extract()?,
            in_race_mode: required_item(data, "in_race_mode")?.extract()?,
            total_racers: required_item(data, "total_racers")?.extract()?,
            course_index: required_item(data, "course_index")?.extract()?,
            course_segment_count: optional_item(data, "course_segment_count", 0)?,
            course_length: optional_item(data, "course_length", 0.0)?,
            player: required_item(data, "player")?.extract()?,
        })
    }

    #[getter]
    fn total_lap_count(&self) -> i32 {
        self.total_lap_count
    }

    #[getter]
    fn difficulty_raw(&self) -> i32 {
        self.difficulty_raw
    }

    #[getter]
    fn difficulty_name(&self) -> &str {
        &self.difficulty_name
    }

    #[getter]
    fn camera_setting_raw(&self) -> i32 {
        self.camera_setting_raw
    }

    #[getter]
    fn camera_setting_name(&self) -> &str {
        &self.camera_setting_name
    }

    #[getter]
    fn race_intro_timer(&self) -> i32 {
        self.race_intro_timer
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
    fn course_segment_count(&self) -> i32 {
        self.course_segment_count
    }

    #[getter]
    fn course_length(&self) -> f32 {
        self.course_length
    }

    #[getter]
    fn player(&self, py: Python<'_>) -> Py<PyPlayerTelemetry> {
        self.player.clone_ref(py)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_lap_count", self.total_lap_count())?;
        dict.set_item("difficulty_raw", self.difficulty_raw())?;
        dict.set_item("difficulty_name", self.difficulty_name())?;
        dict.set_item("camera_setting_raw", self.camera_setting_raw())?;
        dict.set_item("camera_setting_name", self.camera_setting_name())?;
        dict.set_item("race_intro_timer", self.race_intro_timer())?;
        dict.set_item("game_mode_raw", self.game_mode_raw())?;
        dict.set_item("game_mode_name", self.game_mode_name())?;
        dict.set_item("in_race_mode", self.in_race_mode())?;
        dict.set_item("total_racers", self.total_racers())?;
        dict.set_item("course_index", self.course_index())?;
        dict.set_item("course_segment_count", self.course_segment_count())?;
        dict.set_item("course_length", self.course_length())?;
        let player_handle = self.player(py);
        let player = player_handle.bind(py);
        dict.set_item("player", player.call_method0("to_dict")?)?;
        Ok(dict)
    }
}

impl PyTelemetry {
    pub(super) fn from_native(
        telemetry: &TelemetrySnapshot,
        player: Py<PyPlayerTelemetry>,
    ) -> Self {
        Self {
            total_lap_count: telemetry.total_lap_count,
            difficulty_raw: telemetry.difficulty_raw,
            difficulty_name: telemetry.difficulty_name.to_owned(),
            camera_setting_raw: telemetry.camera_setting_raw,
            camera_setting_name: telemetry.camera_setting_name.to_owned(),
            race_intro_timer: telemetry.race_intro_timer,
            game_mode_raw: telemetry.game_mode_raw,
            game_mode_name: telemetry.game_mode_name.to_owned(),
            in_race_mode: telemetry.in_race_mode,
            total_racers: telemetry.total_racers,
            course_index: telemetry.course_index,
            course_segment_count: telemetry.course_segment_count,
            course_length: telemetry.course_length,
            player,
        }
    }
}

fn required_item<'py>(data: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("telemetry snapshot missing {key:?}"))
    })
}

fn optional_item<'py, T>(data: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: pyo3::prelude::FromPyObjectOwned<'py, Error = pyo3::PyErr>,
{
    match data.get_item(key)? {
        Some(value) => value.extract(),
        None => Ok(default),
    }
}
