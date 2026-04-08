// rust/core/game/telemetry/model.rs
//! Public telemetry data structures.

#[derive(Clone, Debug)]
pub struct PlayerTelemetry {
    pub state_flags: u32,
    pub speed_kph: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub boost_timer: i32,
    pub race_distance: f32,
    pub lap_distance: f32,
    pub race_time_ms: i32,
    pub lap: i16,
    pub laps_completed: i16,
    pub position: i32,
}

/// Telemetry snapshot for the current frame, focused on player-one race state.
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    pub game_mode_raw: u32,
    pub game_mode_name: &'static str,
    pub in_race_mode: bool,
    pub course_index: u32,
    pub player: PlayerTelemetry,
}
