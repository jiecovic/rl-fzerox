// rust/core/game/telemetry/model.rs
//! Public telemetry data structures.

const FLAG_RETIRED: u32 = 1 << 18;
const FLAG_FALLING_OFF_TRACK: u32 = 1 << 19;
const FLAG_FINISHED: u32 = 1 << 25;
const FLAG_CRASHED: u32 = 1 << 27;
const FLAG_ACTIVE: u32 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepTelemetrySample {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub race_distance: f32,
    pub reverse_timer: i32,
}

pub(crate) fn terminal_reason_from_state_flags(state_flags: u32) -> Option<&'static str> {
    if (state_flags & FLAG_FINISHED) != 0 {
        return Some("finished");
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
}

/// Telemetry snapshot for the current frame, focused on player-one race state.
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    pub total_lap_count: i32,
    pub difficulty_raw: i32,
    pub difficulty_name: &'static str,
    pub camera_setting_raw: i32,
    pub camera_setting_name: &'static str,
    pub game_mode_raw: u32,
    pub game_mode_name: &'static str,
    pub in_race_mode: bool,
    pub total_racers: i32,
    pub course_index: u32,
    pub player: PlayerTelemetry,
}
