// rust/bindings/emulator/telemetry/root.rs
//! Python-facing root F-Zero X telemetry binding.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bindings::emulator::telemetry::PyPlayerTelemetry;
use crate::bindings::payload::{optional_item, required_item, set_py_dict_items};
use crate::core::telemetry::TelemetrySnapshot;

const TELEMETRY_PAYLOAD: &str = "telemetry snapshot";

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
    menu_selected_mode_raw: i32,
    menu_difficulty_state_raw: i32,
    menu_difficulty_cursor_raw: i32,
    menu_transition_state_raw: i16,
    menu_current_ghost_type_raw: i32,
    queued_game_mode_raw: i32,
    in_race_mode: bool,
    total_racers: i32,
    gp_final_rank: i16,
    gp_points: i16,
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
        telemetry_from_dict(data)
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
    fn menu_selected_mode_raw(&self) -> i32 {
        self.menu_selected_mode_raw
    }

    #[getter]
    fn menu_difficulty_state_raw(&self) -> i32 {
        self.menu_difficulty_state_raw
    }

    #[getter]
    fn menu_difficulty_cursor_raw(&self) -> i32 {
        self.menu_difficulty_cursor_raw
    }

    #[getter]
    fn menu_transition_state_raw(&self) -> i16 {
        self.menu_transition_state_raw
    }

    #[getter]
    fn menu_current_ghost_type_raw(&self) -> i32 {
        self.menu_current_ghost_type_raw
    }

    #[getter]
    fn queued_game_mode_raw(&self) -> i32 {
        self.queued_game_mode_raw
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
    fn gp_final_rank(&self) -> i16 {
        self.gp_final_rank
    }

    #[getter]
    fn gp_points(&self) -> i16 {
        self.gp_points
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
        let player_handle = self.player(py);
        let player = player_handle.bind(py);
        set_py_dict_items!(dict, {
            "total_lap_count" => self.total_lap_count(),
            "difficulty_raw" => self.difficulty_raw(),
            "difficulty_name" => self.difficulty_name(),
            "camera_setting_raw" => self.camera_setting_raw(),
            "camera_setting_name" => self.camera_setting_name(),
            "race_intro_timer" => self.race_intro_timer(),
            "game_mode_raw" => self.game_mode_raw(),
            "game_mode_name" => self.game_mode_name(),
            "menu_selected_mode_raw" => self.menu_selected_mode_raw(),
            "menu_difficulty_state_raw" => self.menu_difficulty_state_raw(),
            "menu_difficulty_cursor_raw" => self.menu_difficulty_cursor_raw(),
            "menu_transition_state_raw" => self.menu_transition_state_raw(),
            "menu_current_ghost_type_raw" => self.menu_current_ghost_type_raw(),
            "queued_game_mode_raw" => self.queued_game_mode_raw(),
            "in_race_mode" => self.in_race_mode(),
            "total_racers" => self.total_racers(),
            "gp_final_rank" => self.gp_final_rank(),
            "gp_points" => self.gp_points(),
            "course_index" => self.course_index(),
            "course_segment_count" => self.course_segment_count(),
            "course_length" => self.course_length(),
            "player" => player.call_method0("to_dict")?,
        })?;
        Ok(dict)
    }
}

fn telemetry_from_dict(data: &Bound<'_, PyDict>) -> PyResult<PyTelemetry> {
    let difficulty_name: Option<String> = optional_item(data, "difficulty_name", None)?;
    let camera_setting_name: Option<String> = optional_item(data, "camera_setting_name", None)?;
    Ok(PyTelemetry {
        total_lap_count: required_item(data, TELEMETRY_PAYLOAD, "total_lap_count")?.extract()?,
        difficulty_raw: optional_item(data, "difficulty_raw", 0)?,
        difficulty_name: difficulty_name.unwrap_or_else(|| "novice".to_owned()),
        camera_setting_raw: optional_item(data, "camera_setting_raw", 2)?,
        camera_setting_name: camera_setting_name.unwrap_or_else(|| "regular".to_owned()),
        race_intro_timer: optional_item(data, "race_intro_timer", 0)?,
        game_mode_raw: required_item(data, TELEMETRY_PAYLOAD, "game_mode_raw")?.extract()?,
        game_mode_name: required_item(data, TELEMETRY_PAYLOAD, "game_mode_name")?.extract()?,
        menu_selected_mode_raw: optional_item(data, "menu_selected_mode_raw", 0)?,
        menu_difficulty_state_raw: optional_item(data, "menu_difficulty_state_raw", 0)?,
        menu_difficulty_cursor_raw: optional_item(data, "menu_difficulty_cursor_raw", 0)?,
        menu_transition_state_raw: optional_item(data, "menu_transition_state_raw", 0)?,
        menu_current_ghost_type_raw: optional_item(data, "menu_current_ghost_type_raw", 0)?,
        queued_game_mode_raw: optional_item(data, "queued_game_mode_raw", 0)?,
        in_race_mode: required_item(data, TELEMETRY_PAYLOAD, "in_race_mode")?.extract()?,
        total_racers: required_item(data, TELEMETRY_PAYLOAD, "total_racers")?.extract()?,
        gp_final_rank: optional_item(data, "gp_final_rank", 0)?,
        gp_points: optional_item(data, "gp_points", 0)?,
        course_index: required_item(data, TELEMETRY_PAYLOAD, "course_index")?.extract()?,
        course_segment_count: optional_item(data, "course_segment_count", 0)?,
        course_length: optional_item(data, "course_length", 0.0)?,
        player: required_item(data, TELEMETRY_PAYLOAD, "player")?.extract()?,
    })
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
            menu_selected_mode_raw: telemetry.menu_selected_mode_raw,
            menu_difficulty_state_raw: telemetry.menu_difficulty_state_raw,
            menu_difficulty_cursor_raw: telemetry.menu_difficulty_cursor_raw,
            menu_transition_state_raw: telemetry.menu_transition_state_raw,
            menu_current_ghost_type_raw: telemetry.menu_current_ghost_type_raw,
            queued_game_mode_raw: telemetry.queued_game_mode_raw,
            in_race_mode: telemetry.in_race_mode,
            total_racers: telemetry.total_racers,
            gp_final_rank: telemetry.gp_final_rank,
            gp_points: telemetry.gp_points,
            course_index: telemetry.course_index,
            course_segment_count: telemetry.course_segment_count,
            course_length: telemetry.course_length,
            player,
        }
    }
}
