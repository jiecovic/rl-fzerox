// rust/core/game/telemetry/model.rs
//! Public telemetry data structures.

#[derive(Clone, Copy, Debug)]
struct RacerStateFlags {
    retired: u32,
    falling_off_track: u32,
    finished: u32,
    crashed: u32,
    active: u32,
    received_damage: u32,
    spinning_out: u32,
}

#[derive(Clone, Copy, Debug)]
struct CourseEffectBits {
    mask: u32,
    pit: u32,
}

const RACER_STATE_FLAGS: RacerStateFlags = RacerStateFlags {
    retired: 1 << 18,
    falling_off_track: 1 << 19,
    finished: 1 << 25,
    crashed: 1 << 27,
    active: 1 << 30,
    received_damage: 1 << 17,
    spinning_out: 1 << 14,
};

const COURSE_EFFECT_BITS: CourseEffectBits = CourseEffectBits { mask: 0xF, pit: 1 };

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CourseEffect {
    None = 0,
    Pit = 1,
    Dirt = 2,
    Dash = 3,
    Ice = 4,
}

impl CourseEffect {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Pit => "pit",
            Self::Dirt => "dirt",
            Self::Dash => "dash",
            Self::Ice => "ice",
        }
    }
}

impl TryFrom<u32> for CourseEffect {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            x if x == Self::None as u32 => Ok(Self::None),
            x if x == Self::Pit as u32 => Ok(Self::Pit),
            x if x == Self::Dirt as u32 => Ok(Self::Dirt),
            x if x == Self::Dash as u32 => Ok(Self::Dash),
            x if x == Self::Ice as u32 => Ok(Self::Ice),
            _ => Err(()),
        }
    }
}

pub fn course_effect_raw_from_state_flags(state_flags: u32) -> u32 {
    state_flags & COURSE_EFFECT_BITS.mask
}

pub fn course_effect_name_from_state_flags(state_flags: u32) -> &'static str {
    CourseEffect::try_from(course_effect_raw_from_state_flags(state_flags))
        .map_or("unknown", CourseEffect::wire_name)
}

pub fn on_energy_refill(state_flags: u32, energy: f32, max_energy: f32) -> bool {
    max_energy > 0.0
        && energy < max_energy
        && course_effect_raw_from_state_flags(state_flags) == COURSE_EFFECT_BITS.pit
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepTelemetrySample {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub race_distance: f32,
    pub reverse_timer: i32,
    pub damage_rumble_counter: i32,
}

impl StepTelemetrySample {
    pub(crate) fn damage_taken(&self) -> bool {
        (self.state_flags & RACER_STATE_FLAGS.received_damage) != 0
            || self.damage_rumble_counter > 0
    }
}

pub(crate) fn terminal_reason_from_state_flags(state_flags: u32) -> Option<&'static str> {
    if (state_flags & RACER_STATE_FLAGS.finished) != 0 {
        return Some("finished");
    }
    if (state_flags & RACER_STATE_FLAGS.spinning_out) != 0 {
        return Some("spinning_out");
    }
    if (state_flags & RACER_STATE_FLAGS.crashed) != 0 {
        return Some("crashed");
    }
    if (state_flags & RACER_STATE_FLAGS.retired) != 0 {
        return Some("retired");
    }
    if (state_flags & RACER_STATE_FLAGS.falling_off_track) != 0 {
        return Some("falling_off_track");
    }
    None
}

#[derive(Clone, Debug)]
pub struct RacerGeometryTelemetry {
    pub segment_index: Option<i32>,
    pub segment_t: f32,
    pub segment_length_proportion: f32,
    pub local_lateral_velocity: f32,
    pub signed_lateral_offset: f32,
    pub lateral_distance: f32,
    pub lateral_displacement_magnitude: f32,
    pub current_radius_left: f32,
    pub current_radius_right: f32,
    pub height_above_ground: f32,
    pub velocity_magnitude: f32,
    pub acceleration_magnitude: f32,
    pub acceleration_force: f32,
    pub drift_attack_force: f32,
    pub collision_mass: f32,
}

impl Default for RacerGeometryTelemetry {
    fn default() -> Self {
        Self {
            segment_index: None,
            segment_t: 0.0,
            segment_length_proportion: 0.0,
            local_lateral_velocity: 0.0,
            signed_lateral_offset: 0.0,
            lateral_distance: 0.0,
            lateral_displacement_magnitude: 0.0,
            current_radius_left: 0.0,
            current_radius_right: 0.0,
            height_above_ground: 0.0,
            velocity_magnitude: 0.0,
            acceleration_magnitude: 0.0,
            acceleration_force: 0.0,
            drift_attack_force: 0.0,
            collision_mass: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayerTelemetry {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub boost_timer: i32,
    pub recoil_tilt_magnitude: f32,
    pub damage_rumble_counter: i32,
    pub reverse_timer: i32,
    pub race_distance: f32,
    pub lap_distance: f32,
    pub race_time_ms: i32,
    pub lap: i16,
    pub laps_completed: i16,
    pub position: i32,
    pub geometry: RacerGeometryTelemetry,
}

impl PlayerTelemetry {
    pub fn terminal_reason(&self) -> Option<&'static str> {
        terminal_reason_from_state_flags(self.state_flags)
    }

    pub fn active(&self) -> bool {
        (self.state_flags & RACER_STATE_FLAGS.active) != 0
    }

    pub fn course_effect_raw(&self) -> u32 {
        course_effect_raw_from_state_flags(self.state_flags)
    }

    pub fn course_effect_name(&self) -> &'static str {
        course_effect_name_from_state_flags(self.state_flags)
    }

    pub fn on_energy_refill(&self) -> bool {
        on_energy_refill(self.state_flags, self.energy, self.max_energy)
    }
}

/// Telemetry snapshot for the current frame, focused on player-one race state.
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    pub total_lap_count: i32,
    pub difficulty_raw: i32,
    pub difficulty_name: &'static str,
    pub camera_setting_raw: i32,
    pub camera_setting_name: &'static str,
    pub race_intro_timer: i32,
    pub game_mode_raw: u32,
    pub game_mode_name: &'static str,
    pub in_race_mode: bool,
    pub total_racers: i32,
    pub course_index: u32,
    pub course_length: f32,
    pub player: PlayerTelemetry,
}
