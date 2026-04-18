// rust/core/game/telemetry/model.rs
//! Public telemetry data structures.

const FLAG_RETIRED: u32 = 1 << 18;
const FLAG_FALLING_OFF_TRACK: u32 = 1 << 19;
const FLAG_FINISHED: u32 = 1 << 25;
const FLAG_CRASHED: u32 = 1 << 27;
const FLAG_ACTIVE: u32 = 1 << 30;
const FLAG_RECEIVED_DAMAGE: u32 = 1 << 17;
const FLAG_SPINNING_OUT: u32 = 1 << 14;
const COURSE_EFFECT_MASK: u32 = 0xF;
const COURSE_EFFECT_PIT: u32 = 1;

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
    state_flags & COURSE_EFFECT_MASK
}

pub fn course_effect_name_from_state_flags(state_flags: u32) -> &'static str {
    CourseEffect::try_from(course_effect_raw_from_state_flags(state_flags))
        .map_or("unknown", CourseEffect::wire_name)
}

pub fn on_energy_refill(state_flags: u32, energy: f32, max_energy: f32) -> bool {
    max_energy > 0.0
        && energy < max_energy
        && course_effect_raw_from_state_flags(state_flags) == COURSE_EFFECT_PIT
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
        (self.state_flags & FLAG_RECEIVED_DAMAGE) != 0 || self.damage_rumble_counter > 0
    }
}

pub(crate) fn terminal_reason_from_state_flags(state_flags: u32) -> Option<&'static str> {
    if (state_flags & FLAG_FINISHED) != 0 {
        return Some("finished");
    }
    if (state_flags & FLAG_SPINNING_OUT) != 0 {
        return Some("spinning_out");
    }
    if (state_flags & FLAG_CRASHED) != 0 {
        return Some("crashed");
    }
    if (state_flags & FLAG_RETIRED) != 0 {
        return Some("retired");
    }
    if (state_flags & FLAG_FALLING_OFF_TRACK) != 0 {
        return Some("falling_off_track");
    }
    None
}

#[derive(Clone, Debug)]
pub struct PlayerTelemetry {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub boost_timer: i32,
    pub recoil_tilt_magnitude: f32,
    pub reverse_timer: i32,
    pub race_distance: f32,
    pub lap_distance: f32,
    pub race_time_ms: i32,
    pub lap: i16,
    pub laps_completed: i16,
    pub position: i32,
}

impl PlayerTelemetry {
    pub fn terminal_reason(&self) -> Option<&'static str> {
        terminal_reason_from_state_flags(self.state_flags)
    }

    pub fn active(&self) -> bool {
        (self.state_flags & FLAG_ACTIVE) != 0
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
    pub player: PlayerTelemetry,
}
