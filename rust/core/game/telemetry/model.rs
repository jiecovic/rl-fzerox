// rust/core/game/telemetry/model.rs
//! Public telemetry data structures.

#[derive(Clone, Copy, Debug)]
pub struct RacerStateFlagSpec {
    pub mask: u32,
    pub label: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct RacerStateFlags {
    pub collision_recoil: u32,
    pub spinning_out: u32,
    pub received_damage: u32,
    pub retired: u32,
    pub falling_off_track: u32,
    pub can_boost: u32,
    pub cpu_controlled: u32,
    pub dash_pad_boost: u32,
    pub finished: u32,
    pub airborne: u32,
    pub crashed: u32,
    pub active: u32,
}

#[derive(Clone, Copy, Debug)]
struct CourseEffectBits {
    mask: u32,
    pit: u32,
}

pub const RACER_STATE_FLAGS: RacerStateFlags = RacerStateFlags {
    collision_recoil: 1 << 13,
    spinning_out: 1 << 14,
    received_damage: 1 << 17,
    retired: 1 << 18,
    falling_off_track: 1 << 19,
    can_boost: 1 << 20,
    cpu_controlled: 1 << 23,
    dash_pad_boost: 1 << 24,
    finished: 1 << 25,
    airborne: 1 << 26,
    crashed: 1 << 27,
    active: 1 << 30,
};

pub const RACER_STATE_FLAG_SPECS: [RacerStateFlagSpec; 11] = [
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.collision_recoil,
        label: "collision_recoil",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.spinning_out,
        label: "spinning_out",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.retired,
        label: "retired",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.falling_off_track,
        label: "falling_off_track",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.can_boost,
        label: "can_boost",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.cpu_controlled,
        label: "cpu_controlled",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.dash_pad_boost,
        label: "dash_pad_boost",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.finished,
        label: "finished",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.airborne,
        label: "airborne",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.crashed,
        label: "crashed",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.active,
        label: "active",
    },
];

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
    pub signed_lateral_offset: f32,
    pub current_radius_left: f32,
    pub current_radius_right: f32,
    pub height_above_ground: f32,
    pub reverse_timer: i32,
    pub damage_rumble_counter: i32,
}

impl StepTelemetrySample {
    pub(crate) fn collision_recoil(&self) -> bool {
        (self.state_flags & RACER_STATE_FLAGS.collision_recoil) != 0
    }

    pub(crate) fn damage_taken(&self) -> bool {
        (self.state_flags & RACER_STATE_FLAGS.received_damage) != 0
            || self.damage_rumble_counter > 0
    }

    pub(crate) fn airborne(&self) -> bool {
        (self.state_flags & RACER_STATE_FLAGS.airborne) != 0
    }

    pub(crate) fn outside_track_bounds(&self) -> bool {
        outside_track_bounds(
            self.signed_lateral_offset,
            self.current_radius_left,
            self.current_radius_right,
        )
    }
}

pub(crate) fn outside_track_bounds(
    offset: f32,
    current_radius_left: f32,
    current_radius_right: f32,
) -> bool {
    // Mirrors the decomp-derived edge threshold used by the Python reward and
    // observation code: 10% beyond the currently active side radius is outside.
    if offset >= 0.0 {
        current_radius_left > 0.0 && offset > current_radius_left * 1.10
    } else {
        current_radius_right > 0.0 && offset < -current_radius_right * 1.10
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

#[derive(Clone, Copy, Debug)]
pub struct RacerGeometryTelemetry {
    pub segment_index: Option<i32>,
    pub segment_t: f32,
    pub segment_length_proportion: f32,
    pub world_pos_x: f32,
    pub world_pos_y: f32,
    pub world_pos_z: f32,
    pub segment_center_x: f32,
    pub segment_center_y: f32,
    pub segment_center_z: f32,
    pub local_lateral_velocity: f32,
    pub signed_lateral_offset: f32,
    pub lateral_distance: f32,
    pub lateral_displacement_magnitude: f32,
    pub current_radius_left: f32,
    pub current_radius_right: f32,
    pub height_above_ground: f32,
    pub future_local_nearest_segment_index: Option<i32>,
    pub future_local_nearest_segment_distance: f32,
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
            world_pos_x: 0.0,
            world_pos_y: 0.0,
            world_pos_z: 0.0,
            segment_center_x: 0.0,
            segment_center_y: 0.0,
            segment_center_z: 0.0,
            local_lateral_velocity: 0.0,
            signed_lateral_offset: 0.0,
            lateral_distance: 0.0,
            lateral_displacement_magnitude: 0.0,
            current_radius_left: 0.0,
            current_radius_right: 0.0,
            height_above_ground: 0.0,
            future_local_nearest_segment_index: None,
            future_local_nearest_segment_distance: 0.0,
            velocity_magnitude: 0.0,
            acceleration_magnitude: 0.0,
            acceleration_force: 0.0,
            drift_attack_force: 0.0,
            collision_mass: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MachineContextTelemetry {
    pub character_index: i16,
    pub body_stat: i8,
    pub boost_stat: i8,
    pub grip_stat: i8,
    pub weight: i16,
    pub engine_setting: f32,
}

impl Default for MachineContextTelemetry {
    fn default() -> Self {
        Self {
            character_index: -1,
            body_stat: 0,
            boost_stat: 0,
            grip_stat: 0,
            weight: 0,
            engine_setting: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlayerTelemetry {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub ko_star_count: i16,
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
    pub machine_context: MachineContextTelemetry,
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
#[derive(Clone, Copy, Debug)]
pub struct TelemetrySnapshot {
    pub total_lap_count: i32,
    pub difficulty_raw: i32,
    pub difficulty_name: &'static str,
    pub camera_setting_raw: i32,
    pub camera_setting_name: &'static str,
    pub race_intro_timer: i32,
    pub game_mode_raw: u32,
    pub game_mode_name: &'static str,
    pub menu_selected_mode_raw: i32,
    pub menu_difficulty_state_raw: i32,
    pub menu_difficulty_cursor_raw: i32,
    pub menu_transition_state_raw: i16,
    pub menu_current_ghost_type_raw: i32,
    pub queued_game_mode_raw: i32,
    pub in_race_mode: bool,
    pub total_racers: i32,
    pub gp_final_rank: i16,
    pub course_index: u32,
    pub course_segment_count: i32,
    pub course_length: f32,
    pub player: PlayerTelemetry,
}
