// rust/bindings/emulator/telemetry.rs
//! PyO3 telemetry objects exposed directly to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bindings::emulator::state::{
    FLAG_ACTIVE, FLAG_AIRBORNE, FLAG_CAN_BOOST, FLAG_COLLISION_RECOIL, FLAG_CPU_CONTROLLED,
    FLAG_CRASHED, FLAG_DASH_PAD_BOOST, FLAG_FALLING_OFF_TRACK, FLAG_FINISHED, FLAG_RETIRED,
    FLAG_SPINNING_OUT, has_state_flag, state_flag_labels,
};
use crate::core::telemetry::{
    MachineContextTelemetry, PlayerTelemetry, RacerGeometryTelemetry, TelemetrySnapshot,
};

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
        recoil_tilt_magnitude,
        reverse_timer,
        race_distance,
        lap_distance,
        race_time_ms,
        lap,
        laps_completed,
        position,
        damage_rumble_counter = 0,
        segment_index = None,
        segment_t = 0.0,
        segment_length_proportion = 0.0,
        local_lateral_velocity = 0.0,
        signed_lateral_offset = 0.0,
        lateral_distance = 0.0,
        lateral_displacement_magnitude = 0.0,
        current_radius_left = 0.0,
        current_radius_right = 0.0,
        height_above_ground = 0.0,
        velocity_magnitude = 0.0,
        acceleration_magnitude = 0.0,
        acceleration_force = 0.0,
        drift_attack_force = 0.0,
        collision_mass = 0.0,
        machine_body_stat = 0,
        machine_boost_stat = 0,
        machine_grip_stat = 0,
        machine_weight = 0,
        engine_setting = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_flags: u32,
        speed_kph: f32,
        energy: f32,
        max_energy: f32,
        boost_timer: i32,
        recoil_tilt_magnitude: f32,
        reverse_timer: i32,
        race_distance: f32,
        lap_distance: f32,
        race_time_ms: i32,
        lap: i16,
        laps_completed: i16,
        position: i32,
        damage_rumble_counter: i32,
        segment_index: Option<i32>,
        segment_t: f32,
        segment_length_proportion: f32,
        local_lateral_velocity: f32,
        signed_lateral_offset: f32,
        lateral_distance: f32,
        lateral_displacement_magnitude: f32,
        current_radius_left: f32,
        current_radius_right: f32,
        height_above_ground: f32,
        velocity_magnitude: f32,
        acceleration_magnitude: f32,
        acceleration_force: f32,
        drift_attack_force: f32,
        collision_mass: f32,
        machine_body_stat: i8,
        machine_boost_stat: i8,
        machine_grip_stat: i8,
        machine_weight: i16,
        engine_setting: f32,
    ) -> Self {
        Self {
            inner: PlayerTelemetry {
                state_flags,
                speed_kph,
                energy,
                max_energy,
                boost_timer,
                recoil_tilt_magnitude,
                damage_rumble_counter,
                reverse_timer,
                race_distance,
                lap_distance,
                race_time_ms,
                lap,
                laps_completed,
                position,
                geometry: RacerGeometryTelemetry {
                    segment_index,
                    segment_t,
                    segment_length_proportion,
                    local_lateral_velocity,
                    signed_lateral_offset,
                    lateral_distance,
                    lateral_displacement_magnitude,
                    current_radius_left,
                    current_radius_right,
                    height_above_ground,
                    velocity_magnitude,
                    acceleration_magnitude,
                    acceleration_force,
                    drift_attack_force,
                    collision_mass,
                },
                machine_context: MachineContextTelemetry {
                    body_stat: machine_body_stat,
                    boost_stat: machine_boost_stat,
                    grip_stat: machine_grip_stat,
                    weight: machine_weight,
                    engine_setting,
                },
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
    fn recoil_tilt_magnitude(&self) -> f32 {
        self.inner.recoil_tilt_magnitude
    }

    #[getter]
    fn damage_rumble_counter(&self) -> i32 {
        self.inner.damage_rumble_counter
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
    fn segment_index(&self) -> Option<i32> {
        self.inner.geometry.segment_index
    }

    #[getter]
    fn segment_t(&self) -> f32 {
        self.inner.geometry.segment_t
    }

    #[getter]
    fn segment_length_proportion(&self) -> f32 {
        self.inner.geometry.segment_length_proportion
    }

    #[getter]
    fn local_lateral_velocity(&self) -> f32 {
        self.inner.geometry.local_lateral_velocity
    }

    #[getter]
    fn signed_lateral_offset(&self) -> f32 {
        self.inner.geometry.signed_lateral_offset
    }

    #[getter]
    fn lateral_distance(&self) -> f32 {
        self.inner.geometry.lateral_distance
    }

    #[getter]
    fn lateral_displacement_magnitude(&self) -> f32 {
        self.inner.geometry.lateral_displacement_magnitude
    }

    #[getter]
    fn current_radius_left(&self) -> f32 {
        self.inner.geometry.current_radius_left
    }

    #[getter]
    fn current_radius_right(&self) -> f32 {
        self.inner.geometry.current_radius_right
    }

    #[getter]
    fn height_above_ground(&self) -> f32 {
        self.inner.geometry.height_above_ground
    }

    #[getter]
    fn velocity_magnitude(&self) -> f32 {
        self.inner.geometry.velocity_magnitude
    }

    #[getter]
    fn acceleration_magnitude(&self) -> f32 {
        self.inner.geometry.acceleration_magnitude
    }

    #[getter]
    fn acceleration_force(&self) -> f32 {
        self.inner.geometry.acceleration_force
    }

    #[getter]
    fn drift_attack_force(&self) -> f32 {
        self.inner.geometry.drift_attack_force
    }

    #[getter]
    fn collision_mass(&self) -> f32 {
        self.inner.geometry.collision_mass
    }

    #[getter]
    fn machine_body_stat(&self) -> i8 {
        self.inner.machine_context.body_stat
    }

    #[getter]
    fn machine_boost_stat(&self) -> i8 {
        self.inner.machine_context.boost_stat
    }

    #[getter]
    fn machine_grip_stat(&self) -> i8 {
        self.inner.machine_context.grip_stat
    }

    #[getter]
    fn machine_weight(&self) -> i16 {
        self.inner.machine_context.weight
    }

    #[getter]
    fn engine_setting(&self) -> f32 {
        self.inner.machine_context.engine_setting
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
    fn course_effect_raw(&self) -> u32 {
        self.inner.course_effect_raw()
    }

    #[getter]
    fn course_effect_name(&self) -> &'static str {
        self.inner.course_effect_name()
    }

    #[getter]
    fn on_energy_refill(&self) -> bool {
        self.inner.on_energy_refill()
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
        dict.set_item("recoil_tilt_magnitude", self.recoil_tilt_magnitude())?;
        dict.set_item("damage_rumble_counter", self.damage_rumble_counter())?;
        dict.set_item("reverse_timer", self.reverse_timer())?;
        dict.set_item("race_distance", self.race_distance())?;
        dict.set_item("lap_distance", self.lap_distance())?;
        dict.set_item("race_time_ms", self.race_time_ms())?;
        dict.set_item("lap", self.lap())?;
        dict.set_item("laps_completed", self.laps_completed())?;
        dict.set_item("position", self.position())?;
        dict.set_item("segment_index", self.segment_index())?;
        dict.set_item("segment_t", self.segment_t())?;
        dict.set_item(
            "segment_length_proportion",
            self.segment_length_proportion(),
        )?;
        dict.set_item("local_lateral_velocity", self.local_lateral_velocity())?;
        dict.set_item("signed_lateral_offset", self.signed_lateral_offset())?;
        dict.set_item("lateral_distance", self.lateral_distance())?;
        dict.set_item(
            "lateral_displacement_magnitude",
            self.lateral_displacement_magnitude(),
        )?;
        dict.set_item("current_radius_left", self.current_radius_left())?;
        dict.set_item("current_radius_right", self.current_radius_right())?;
        dict.set_item("height_above_ground", self.height_above_ground())?;
        dict.set_item("velocity_magnitude", self.velocity_magnitude())?;
        dict.set_item("acceleration_magnitude", self.acceleration_magnitude())?;
        dict.set_item("acceleration_force", self.acceleration_force())?;
        dict.set_item("drift_attack_force", self.drift_attack_force())?;
        dict.set_item("collision_mass", self.collision_mass())?;
        dict.set_item("machine_body_stat", self.machine_body_stat())?;
        dict.set_item("machine_boost_stat", self.machine_boost_stat())?;
        dict.set_item("machine_grip_stat", self.machine_grip_stat())?;
        dict.set_item("machine_weight", self.machine_weight())?;
        dict.set_item("engine_setting", self.engine_setting())?;
        dict.set_item("course_effect_raw", self.course_effect_raw())?;
        dict.set_item("course_effect_name", self.course_effect_name())?;
        dict.set_item("on_energy_refill", self.on_energy_refill())?;
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
    course_length: f32,
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
        course_length = 0.0,
        difficulty_raw = 0,
        difficulty_name = None,
        camera_setting_raw = 2,
        camera_setting_name = None,
        race_intro_timer = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        total_lap_count: i32,
        game_mode_raw: u32,
        game_mode_name: String,
        in_race_mode: bool,
        total_racers: i32,
        course_index: u32,
        player: Py<PyPlayerTelemetry>,
        course_length: f32,
        difficulty_raw: i32,
        difficulty_name: Option<String>,
        camera_setting_raw: i32,
        camera_setting_name: Option<String>,
        race_intro_timer: i32,
    ) -> Self {
        Self {
            total_lap_count,
            difficulty_raw,
            difficulty_name: difficulty_name.unwrap_or_else(|| "novice".to_owned()),
            camera_setting_raw,
            camera_setting_name: camera_setting_name.unwrap_or_else(|| "regular".to_owned()),
            race_intro_timer,
            game_mode_raw,
            game_mode_name,
            in_race_mode,
            total_racers,
            course_index,
            course_length,
            player,
        }
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
        dict.set_item("course_length", self.course_length())?;
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
            course_length: telemetry.course_length,
            player,
        },
    )
}
