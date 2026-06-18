// rust/bindings/emulator/telemetry/player.rs
//! Python-facing player telemetry binding.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bindings::emulator::state::{RACER_STATE_FLAGS, has_state_flag, state_flag_labels};
use crate::bindings::payload::{optional_item, required_item, set_py_dict_items};
use crate::core::telemetry::{MachineContextTelemetry, PlayerTelemetry, RacerGeometryTelemetry};

const PLAYER_TELEMETRY_PAYLOAD: &str = "player telemetry";

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
    #[pyo3(signature = (data))]
    fn new(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: PlayerTelemetry {
                state_flags: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "state_flags")?
                    .extract()?,
                speed_kph: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "speed_kph")?.extract()?,
                energy: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "energy")?.extract()?,
                max_energy: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "max_energy")?
                    .extract()?,
                ko_star_count: optional_item(data, "ko_star_count", 0)?,
                boost_timer: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "boost_timer")?
                    .extract()?,
                recoil_tilt_magnitude: required_item(
                    data,
                    PLAYER_TELEMETRY_PAYLOAD,
                    "recoil_tilt_magnitude",
                )?
                .extract()?,
                damage_rumble_counter: optional_item(data, "damage_rumble_counter", 0)?,
                reverse_timer: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "reverse_timer")?
                    .extract()?,
                race_distance: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "race_distance")?
                    .extract()?,
                lap_distance: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "lap_distance")?
                    .extract()?,
                race_time_ms: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "race_time_ms")?
                    .extract()?,
                lap: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "lap")?.extract()?,
                laps_completed: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "laps_completed")?
                    .extract()?,
                position: required_item(data, PLAYER_TELEMETRY_PAYLOAD, "position")?.extract()?,
                geometry: RacerGeometryTelemetry {
                    segment_index: optional_item(data, "segment_index", None)?,
                    segment_t: optional_item(data, "segment_t", 0.0)?,
                    segment_length_proportion: optional_item(
                        data,
                        "segment_length_proportion",
                        0.0,
                    )?,
                    world_pos_x: optional_item(data, "world_pos_x", 0.0)?,
                    world_pos_y: optional_item(data, "world_pos_y", 0.0)?,
                    world_pos_z: optional_item(data, "world_pos_z", 0.0)?,
                    segment_center_x: optional_item(data, "segment_center_x", 0.0)?,
                    segment_center_y: optional_item(data, "segment_center_y", 0.0)?,
                    segment_center_z: optional_item(data, "segment_center_z", 0.0)?,
                    local_lateral_velocity: optional_item(data, "local_lateral_velocity", 0.0)?,
                    signed_lateral_offset: optional_item(data, "signed_lateral_offset", 0.0)?,
                    lateral_distance: optional_item(data, "lateral_distance", 0.0)?,
                    lateral_displacement_magnitude: optional_item(
                        data,
                        "lateral_displacement_magnitude",
                        0.0,
                    )?,
                    current_radius_left: optional_item(data, "current_radius_left", 0.0)?,
                    current_radius_right: optional_item(data, "current_radius_right", 0.0)?,
                    height_above_ground: optional_item(data, "height_above_ground", 0.0)?,
                    future_local_nearest_segment_index: optional_item(
                        data,
                        "future_local_nearest_segment_index",
                        None,
                    )?,
                    future_local_nearest_segment_distance: optional_item(
                        data,
                        "future_local_nearest_segment_distance",
                        0.0,
                    )?,
                    velocity_magnitude: optional_item(data, "velocity_magnitude", 0.0)?,
                    acceleration_magnitude: optional_item(data, "acceleration_magnitude", 0.0)?,
                    acceleration_force: optional_item(data, "acceleration_force", 0.0)?,
                    drift_attack_force: optional_item(data, "drift_attack_force", 0.0)?,
                    collision_mass: optional_item(data, "collision_mass", 0.0)?,
                },
                machine_context: MachineContextTelemetry {
                    character_index: optional_item(data, "machine_character_index", -1)?,
                    body_stat: optional_item(data, "machine_body_stat", 0)?,
                    boost_stat: optional_item(data, "machine_boost_stat", 0)?,
                    grip_stat: optional_item(data, "machine_grip_stat", 0)?,
                    weight: optional_item(data, "machine_weight", 0)?,
                    engine_setting: optional_item(data, "engine_setting", 0.0)?,
                },
            },
        })
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
    fn ko_star_count(&self) -> i16 {
        self.inner.ko_star_count
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
    fn world_pos_x(&self) -> f32 {
        self.inner.geometry.world_pos_x
    }

    #[getter]
    fn world_pos_y(&self) -> f32 {
        self.inner.geometry.world_pos_y
    }

    #[getter]
    fn world_pos_z(&self) -> f32 {
        self.inner.geometry.world_pos_z
    }

    #[getter]
    fn segment_center_x(&self) -> f32 {
        self.inner.geometry.segment_center_x
    }

    #[getter]
    fn segment_center_y(&self) -> f32 {
        self.inner.geometry.segment_center_y
    }

    #[getter]
    fn segment_center_z(&self) -> f32 {
        self.inner.geometry.segment_center_z
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
    fn future_local_nearest_segment_index(&self) -> Option<i32> {
        self.inner.geometry.future_local_nearest_segment_index
    }

    #[getter]
    fn future_local_nearest_segment_distance(&self) -> f32 {
        self.inner.geometry.future_local_nearest_segment_distance
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
    fn machine_character_index(&self) -> i16 {
        self.inner.machine_context.character_index
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
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.collision_recoil)
    }

    #[getter]
    fn spinning_out(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.spinning_out)
    }

    #[getter]
    fn retired(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.retired)
    }

    #[getter]
    fn falling_off_track(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.falling_off_track)
    }

    #[getter]
    fn can_boost(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.can_boost)
    }

    #[getter]
    fn cpu_controlled(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.cpu_controlled)
    }

    #[getter]
    fn dash_pad_boost(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.dash_pad_boost)
    }

    #[getter]
    fn finished(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.finished)
    }

    #[getter]
    fn airborne(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.airborne)
    }

    #[getter]
    fn crashed(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.crashed)
    }

    #[getter]
    fn active(&self) -> bool {
        has_state_flag(self.inner.state_flags, RACER_STATE_FLAGS.active)
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
        set_py_dict_items!(dict, {
            "state_flags" => self.state_flags(),
            "state_labels" => self.state_labels(py)?,
            "speed_kph" => self.speed_kph(),
            "energy" => self.energy(),
            "max_energy" => self.max_energy(),
            "ko_star_count" => self.ko_star_count(),
            "boost_timer" => self.boost_timer(),
            "recoil_tilt_magnitude" => self.recoil_tilt_magnitude(),
            "damage_rumble_counter" => self.damage_rumble_counter(),
            "reverse_timer" => self.reverse_timer(),
            "race_distance" => self.race_distance(),
            "lap_distance" => self.lap_distance(),
            "race_time_ms" => self.race_time_ms(),
            "lap" => self.lap(),
            "laps_completed" => self.laps_completed(),
            "position" => self.position(),
            "segment_index" => self.segment_index(),
            "segment_t" => self.segment_t(),
            "segment_length_proportion" => self.segment_length_proportion(),
            "world_pos_x" => self.world_pos_x(),
            "world_pos_y" => self.world_pos_y(),
            "world_pos_z" => self.world_pos_z(),
            "segment_center_x" => self.segment_center_x(),
            "segment_center_y" => self.segment_center_y(),
            "segment_center_z" => self.segment_center_z(),
            "local_lateral_velocity" => self.local_lateral_velocity(),
            "signed_lateral_offset" => self.signed_lateral_offset(),
            "lateral_distance" => self.lateral_distance(),
            "lateral_displacement_magnitude" => self.lateral_displacement_magnitude(),
            "current_radius_left" => self.current_radius_left(),
            "current_radius_right" => self.current_radius_right(),
            "height_above_ground" => self.height_above_ground(),
            "future_local_nearest_segment_index" =>
                self.future_local_nearest_segment_index(),
            "future_local_nearest_segment_distance" =>
                self.future_local_nearest_segment_distance(),
            "velocity_magnitude" => self.velocity_magnitude(),
            "acceleration_magnitude" => self.acceleration_magnitude(),
            "acceleration_force" => self.acceleration_force(),
            "drift_attack_force" => self.drift_attack_force(),
            "collision_mass" => self.collision_mass(),
            "machine_character_index" => self.machine_character_index(),
            "machine_body_stat" => self.machine_body_stat(),
            "machine_boost_stat" => self.machine_boost_stat(),
            "machine_grip_stat" => self.machine_grip_stat(),
            "machine_weight" => self.machine_weight(),
            "engine_setting" => self.engine_setting(),
            "course_effect_raw" => self.course_effect_raw(),
            "course_effect_name" => self.course_effect_name(),
            "on_energy_refill" => self.on_energy_refill(),
        })?;
        Ok(dict)
    }
}

impl PyPlayerTelemetry {
    pub(super) fn from_native(player: &PlayerTelemetry) -> Self {
        Self { inner: *player }
    }
}
